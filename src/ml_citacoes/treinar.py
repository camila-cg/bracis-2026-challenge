"""Treino LoRA reproduzível para um fold ou para todos os documentos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .aumento import aumentar_citacao_real
from .bio import bio_para_spans, combinar_logits_janelas, spans_para_bio
from .contratos import ID2LABEL, LABEL2ID


def _dependencias():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("treino requer o grupo ml; execute: uv sync --group ml") from exc
    return locals()


class ListaDataset:
    def __init__(self, itens: list[dict]):
        self.itens = itens

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, indice):
        return self.itens[indice]


def _equilibrar_janelas(exemplos: list[dict], proporcao_negativas: float, seed: int) -> tuple[list[dict], dict]:
    """Limita janelas só com O sem alterar a distribuição da validação."""
    positivas = [item for item in exemplos if any(rotulo > 0 for rotulo in item["labels"])]
    negativas = [item for item in exemplos if not any(rotulo > 0 for rotulo in item["labels"])]
    if not positivas:
        raise ValueError("treino sem janelas que contenham citações")
    limite = min(len(negativas), math.ceil(len(positivas) * proporcao_negativas))
    rng = random.Random(seed)
    negativas_amostradas = rng.sample(negativas, limite) if limite < len(negativas) else negativas
    equilibradas = positivas + negativas_amostradas
    rng.shuffle(equilibradas)
    return equilibradas, {
        "janelas_positivas": len(positivas),
        "janelas_negativas_disponiveis": len(negativas),
        "janelas_negativas_usadas": len(negativas_amostradas),
    }


def _pesos_classes(exemplos: list[dict], potencia: float = 0.5, peso_maximo: float = 10.0) -> tuple[list[float], dict]:
    """Calcula pesos relativos a O para a CrossEntropy, com teto de estabilidade."""
    contagens = Counter(
        rotulo
        for item in exemplos
        for rotulo in item["labels"]
        if rotulo != -100
    )
    if contagens.get(0, 0) == 0:
        raise ValueError("treino sem tokens O")
    ausentes = [rotulo for rotulo in range(len(LABEL2ID)) if contagens.get(rotulo, 0) == 0]
    if ausentes:
        raise ValueError(f"treino sem exemplos dos rótulos: {ausentes}")
    referencia = contagens[0]
    pesos = [min(peso_maximo, (referencia / contagens[i]) ** potencia) for i in range(len(LABEL2ID))]
    return pesos, {ID2LABEL[i]: contagens[i] for i in range(len(LABEL2ID))}


def _amostrar_tokens_o(exemplos: list[dict], proporcao_o: float, seed: int) -> tuple[list[dict], dict]:
    """Ignora parte de O somente na perda; os tokens continuam no contexto do BERT."""
    positivos = sum(
        rotulo > 0
        for item in exemplos
        for rotulo in item["labels"]
        if rotulo != -100
    )
    if positivos == 0:
        raise ValueError("treino sem tokens positivos")
    posicoes_o = [
        (indice_item, indice_rotulo)
        for indice_item, item in enumerate(exemplos)
        for indice_rotulo, rotulo in enumerate(item["labels"])
        if rotulo == 0
    ]
    limite = min(len(posicoes_o), math.ceil(positivos * proporcao_o))
    rng = random.Random(seed)

    # Uma janela inteiramente negativa precisa conservar ao menos um alvo para
    # que nenhum lote produza CrossEntropy sobre apenas índices ignorados.
    obrigatorias = set()
    for indice_item, item in enumerate(exemplos):
        if not any(rotulo > 0 for rotulo in item["labels"]):
            candidatas = [
                (indice_item, indice_rotulo)
                for indice_rotulo, rotulo in enumerate(item["labels"])
                if rotulo == 0
            ]
            if candidatas:
                obrigatorias.add(rng.choice(candidatas))
    restantes = [posicao for posicao in posicoes_o if posicao not in obrigatorias]
    quantidade_extra = max(0, limite - len(obrigatorias))
    escolhidas = obrigatorias | set(
        rng.sample(restantes, quantidade_extra)
        if quantidade_extra < len(restantes) else restantes
    )

    saida = []
    for indice_item, item in enumerate(exemplos):
        labels = [
            rotulo if rotulo != 0 or (indice_item, indice_rotulo) in escolhidas else -100
            for indice_rotulo, rotulo in enumerate(item["labels"])
        ]
        saida.append({**item, "labels": labels})
    return saida, {
        "tokens_positivos": positivos,
        "tokens_o_disponiveis": len(posicoes_o),
        "tokens_o_usados_na_perda": len(escolhidas),
        "proporcao_o_por_positivo": proporcao_o,
    }


def _metricas_tokens(eval_pred) -> dict[str, float]:
    """Expõe explicitamente se o modelo voltou a colapsar para O."""
    import numpy as np

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    previstos = np.argmax(logits, axis=-1)
    mascara = labels != -100
    verdade_positiva = (labels != 0) & mascara
    previsto_positivo = (previstos != 0) & mascara
    corretos_positivos = (previstos == labels) & verdade_positiva
    tp = int(corretos_positivos.sum())
    fp = int(previsto_positivo.sum()) - tp
    fn = int(verdade_positiva.sum()) - tp
    precisao = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    f2 = 5 * precisao * recall / (4 * precisao + recall) if precisao + recall else 0.0
    tp_deteccao = int((verdade_positiva & previsto_positivo).sum())
    precisao_deteccao = tp_deteccao / int(previsto_positivo.sum()) if previsto_positivo.any() else 0.0
    recall_deteccao = tp_deteccao / int(verdade_positiva.sum()) if verdade_positiva.any() else 0.0
    return {
        "positive_precision": precisao,
        "positive_recall": recall,
        "positive_f1": f1,
        "positive_f2": f2,
        "detection_precision": precisao_deteccao,
        "detection_recall": recall_deteccao,
    }


def _argumentos_checkpointing(config: dict) -> dict:
    """Configura recomputação de ativações sem depender de torch nos testes."""
    ativo = bool(config.get("gradient_checkpointing", False))
    argumentos = {"gradient_checkpointing": ativo}
    if ativo:
        # O modo não reentrante funciona quando o backbone está congelado pelo
        # LoRA e evita exigir gradiente nos embeddings de entrada.
        argumentos["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return argumentos


def _ler_jsonl(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def _aumentar_documento(documento: dict, seed: int) -> tuple[dict, list[dict]]:
    """Trocas 1:1 somente em citações reais; offsets permanecem invariantes."""
    texto = list(documento["texto"])
    registro = []
    for indice, span in enumerate(documento["spans"]):
        if span.get("classificacao") != "real":
            continue
        original = "".join(texto[span["inicio"]:span["fim"]])
        alterado, transformacoes = aumentar_citacao_real(original, seed=seed + indice)
        if len(alterado) != len(original):
            # "m" -> "rn" é seguro isoladamente, mas não é usado aqui porque
            # mudaria todos os offsets seguintes do documento.
            continue
        texto[span["inicio"]:span["fim"]] = alterado
        registro.extend({"span": indice, **asdict(t)} for t in transformacoes)
    return {**documento, "texto": "".join(texto)}, registro


def _tokenizar(documentos: list[dict], tokenizer, max_length: int, stride: int) -> list[dict]:
    exemplos = []
    for documento in documentos:
        codificado = tokenizer(
            documento["texto"], truncation=True, max_length=max_length, stride=stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
        )
        offsets_janelas = codificado.pop("offset_mapping")
        codificado.pop("overflow_to_sample_mapping", None)
        for indice, offsets in enumerate(offsets_janelas):
            item = {chave: valores[indice] for chave, valores in codificado.items()}
            item["labels"] = spans_para_bio(offsets, documento["spans"])
            exemplos.append(item)
    return exemplos


def _prever(modelo, tokenizer, texto: str, max_length: int, stride: int, torch) -> list[dict]:
    lote = tokenizer(
        texto, truncation=True, max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        return_tensors="pt", padding=True,
    )
    offsets = lote.pop("offset_mapping").tolist()
    lote.pop("overflow_to_sample_mapping", None)
    dispositivo = next(modelo.parameters()).device
    with torch.no_grad():
        logits = modelo(**{k: v.to(dispositivo) for k, v in lote.items()}).logits.detach().cpu().tolist()
    offsets, labels, scores = combinar_logits_janelas(offsets, logits)
    spans, diagnostico = bio_para_spans(labels, offsets, scores)
    return [
        {"inicio": s.inicio, "fim": s.fim, "tipo": s.tipo, "score": s.score_bruto}
        for s in spans
    ], diagnostico


def treinar(args) -> None:
    deps = _dependencias()
    torch = deps["torch"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    documentos = _ler_jsonl(args.dados / "documentos.jsonl")
    manifesto_folds = json.loads((args.dados / "folds.json").read_text(encoding="utf-8"))
    if args.fold == "todos":
        treino, validacao = documentos, []
    else:
        fold = manifesto_folds["folds"][int(args.fold)]
        ids_treino, ids_validacao = set(fold["treino"]), set(fold["validacao"])
        if ids_treino & ids_validacao:
            raise ValueError("vazamento entre treino e validação")
        treino = [doc for doc in documentos if doc["documento_id"] in ids_treino]
        validacao = [doc for doc in documentos if doc["documento_id"] in ids_validacao]
        encontrados_treino = {doc["documento_id"] for doc in treino}
        encontrados_validacao = {doc["documento_id"] for doc in validacao}
        if encontrados_treino != ids_treino or encontrados_validacao != ids_validacao:
            raise ValueError("fold referencia documentos ausentes de documentos.jsonl")
        if any(doc.get("origem") == "sintetico" for doc in validacao):
            raise ValueError("documentos sintéticos não podem entrar na validação")

    seed = int(config.get("seed", manifesto_folds["seed"]))
    deps["set_seed"](seed)
    random.seed(seed)
    checkpoint = config["checkpoint"]
    revisao_pedida = config.get("revision") or "main"
    tokenizer = deps["AutoTokenizer"].from_pretrained(
        checkpoint, revision=revisao_pedida, local_files_only=not args.permitir_download, use_fast=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("offset_mapping requer tokenizer fast")
    modelo = deps["AutoModelForTokenClassification"].from_pretrained(
        checkpoint, revision=revisao_pedida, local_files_only=not args.permitir_download,
        num_labels=5, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    revisao_resolvida = getattr(modelo.config, "_commit_hash", None)
    if not revisao_resolvida:
        raise ValueError("não foi possível resolver a revisão exata do checkpoint")
    lora = deps["LoraConfig"](
        task_type=deps["TaskType"].TOKEN_CLS,
        r=int(config.get("rank", 7)),
        lora_alpha=int(config.get("alpha", 14)),
        lora_dropout=float(config.get("dropout", 0.075)),
        target_modules=list(config.get("target_modules", ["query", "value"])),
        bias="none",
        modules_to_save=["classifier"],
    )
    modelo = deps["get_peft_model"](modelo, lora)
    max_length, stride = int(config.get("max_length", 512)), int(config.get("stride", 96))
    transformacoes = []
    treino_aumentado = []
    if config.get("ocr_augmentation", True):
        for indice, documento in enumerate(treino):
            if documento.get("origem") == "sintetico" and not config.get("ocr_augmentation_synthetic", False):
                continue
            aumentado, log = _aumentar_documento(documento, seed + 10_000 + indice)
            if log:
                treino_aumentado.append(aumentado)
                transformacoes.append({"documento_id": documento["documento_id"], "transformacoes": log})
    exemplos_treino = _tokenizar(treino + treino_aumentado, tokenizer, max_length, stride)
    exemplos_validacao = _tokenizar(validacao, tokenizer, max_length, stride) if validacao else []
    proporcao_negativas = float(config.get("negative_window_ratio", 1.0))
    exemplos_treino, amostragem = _equilibrar_janelas(exemplos_treino, proporcao_negativas, seed)
    exemplos_treino, amostragem_tokens = _amostrar_tokens_o(
        exemplos_treino,
        proporcao_o=float(config.get("o_token_ratio", 3.0)),
        seed=seed + 1,
    )
    pesos_classes, contagens_rotulos = _pesos_classes(
        exemplos_treino,
        potencia=float(config.get("class_weight_power", 0.5)),
        peso_maximo=float(config.get("class_weight_max", 10.0)),
    )
    args.saida.mkdir(parents=True, exist_ok=True)
    metrica_checkpoint = str(config.get("metric_for_best_model", "positive_f1"))
    if metrica_checkpoint not in {"positive_f1", "positive_f2", "positive_recall"}:
        raise ValueError("metric_for_best_model deve ser positive_f1, positive_f2 ou positive_recall")
    argumentos = deps["TrainingArguments"](
        output_dir=str(args.saida / "checkpoints"),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(config.get("batch_size", 8)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", config.get("batch_size", 8))),
        gradient_accumulation_steps=int(config.get("gradient_accumulation", 2)),
        num_train_epochs=float(config.get("epochs", 8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        eval_strategy="epoch" if validacao else "no",
        save_strategy="epoch" if validacao else "no",
        load_best_model_at_end=bool(validacao),
        metric_for_best_model=metrica_checkpoint if validacao else None,
        greater_is_better=True if validacao else None,
        seed=seed,
        data_seed=seed,
        report_to=[],
        save_total_limit=2,
        **_argumentos_checkpointing(config),
    )
    class TrainerPonderado(deps["Trainer"]):
        def __init__(self, *trainer_args, pesos, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.pesos = torch.tensor(pesos, dtype=torch.float32)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            entradas = dict(inputs)
            labels = entradas.pop("labels")
            outputs = model(**entradas)
            logits = outputs.logits
            funcao_perda = torch.nn.CrossEntropyLoss(
                weight=self.pesos.to(logits.device), ignore_index=-100
            )
            loss = funcao_perda(logits.reshape(-1, model.config.num_labels), labels.reshape(-1))
            return (loss, outputs) if return_outputs else loss

    treinador = TrainerPonderado(
        model=modelo,
        args=argumentos,
        train_dataset=ListaDataset(exemplos_treino),
        eval_dataset=ListaDataset(exemplos_validacao) if validacao else None,
        data_collator=deps["DataCollatorForTokenClassification"](tokenizer),
        processing_class=tokenizer,
        compute_metrics=_metricas_tokens if validacao else None,
        pesos=pesos_classes,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inicio_treino = time.perf_counter()
    treinador.train()
    duracao_treino = time.perf_counter() - inicio_treino
    pico_alocado_mib = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else None
    )
    pico_reservado_mib = (
        torch.cuda.max_memory_reserved() / (1024 ** 2)
        if torch.cuda.is_available() else None
    )
    modelo.save_pretrained(args.saida / "adaptador")
    tokenizer.save_pretrained(args.saida / "tokenizer")
    manifesto = {
        "checkpoint": checkpoint,
        "revisao_solicitada": revisao_pedida,
        "revisao_resolvida": revisao_resolvida,
        "labels": list(LABEL2ID),
        "lora": {"rank": lora.r, "alpha": lora.lora_alpha, "dropout": lora.lora_dropout, "target_modules": list(lora.target_modules)},
        "max_length": max_length,
        "stride": stride,
        "seed": seed,
        "fold": args.fold,
        "documentos_treino": [doc["documento_id"] for doc in treino],
        "documentos_treino_originais": [doc["documento_id"] for doc in treino if doc.get("origem") != "sintetico"],
        "documentos_treino_sinteticos": [doc["documento_id"] for doc in treino if doc.get("origem") == "sintetico"],
        "documentos_validacao": [doc["documento_id"] for doc in validacao],
        "dataset_sha256": hashlib.sha256((args.dados / "documentos.jsonl").read_bytes()).hexdigest(),
        "temperatura": 1.0,
        "calibracao": "nenhuma",
        "metrica_selecao_checkpoint": metrica_checkpoint,
        "recursos_treino": {
            "batch_size": int(config.get("batch_size", 8)),
            "eval_batch_size": int(config.get("eval_batch_size", config.get("batch_size", 8))),
            "gradient_accumulation": int(config.get("gradient_accumulation", 2)),
            "batch_efetivo": int(config.get("batch_size", 8)) * int(config.get("gradient_accumulation", 2)),
            "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
            "precisao": "fp32",
            "duracao_segundos": duracao_treino,
            "pico_vram_alocada_mib": pico_alocado_mib,
            "pico_vram_reservada_mib": pico_reservado_mib,
        },
        "aumentos_ocr": transformacoes,
        "balanceamento": {
            **amostragem,
            **amostragem_tokens,
            "proporcao_negativas": proporcao_negativas,
            "contagens_rotulos": contagens_rotulos,
            "pesos_classes": {ID2LABEL[i]: peso for i, peso in enumerate(pesos_classes)},
        },
    }
    (args.saida / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    if validacao:
        predicoes, invalidas, fragmentos, documentos_invalidos = {}, 0, 0, 0
        modelo.eval()
        for documento in validacao:
            spans, diag = _prever(modelo, tokenizer, documento["texto"], max_length, stride, torch)
            predicoes[documento["documento_id"]] = spans
            invalidas += diag.invalidas
            fragmentos += diag.fragmentos
            documentos_invalidos += int(diag.invalidas > 0)
        (args.saida / "predicoes_validacao.json").write_text(json.dumps(predicoes, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.saida / "diagnostico_bio.json").write_text(json.dumps({
            "sequencias_invalidas": invalidas,
            "spans_fragmentados": fragmentos,
            "documentos_com_bio_invalido": documentos_invalidos,
            "total_documentos": len(validacao),
            "total_spans_preditos": sum(len(itens) for itens in predicoes.values()),
        }, indent=2), encoding="utf-8")
        if not any(predicoes.values()):
            raise RuntimeError(
                "modelo colapsou para O: nenhuma citação foi prevista na validação; "
                "não use este fold nem prossiga para a seleção"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", choices=["0", "1", "2", "3", "4", "todos"], required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--permitir-download", action="store_true", help="autoriza baixar o checkpoint ao executar este comando")
    treinar(parser.parse_args())


if __name__ == "__main__":
    main()
