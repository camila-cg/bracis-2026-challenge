"""Smoke test de um fold: LoRA de alto recall seguido de verificador LLM zero-shot."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from .avaliar_validacao import _com_documento, _detalhar_erros
from .experimento_bio import _metricas, validar_entrada
from .fusao import fusionar
from .inferencia import DetectorBIO
from .metricas import avaliar_spans
from .verificador_llm import PROMPT_SHA256, PROMPT_VERSAO, VerificadorLlama, VerificadorOllama
from steps.extrair_candidatos import extrair
from steps.filtrar_distratores import e_distrator, filtrar_distratores


class _MedidorVRAM:
    """Amostra a VRAM de todos os processos; inclui o servidor Ollama."""

    def __init__(self, intervalo: float = 0.5):
        self.intervalo = intervalo
        self.pico_mib: int | None = None
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._executar, daemon=True)

    @staticmethod
    def _ler() -> int | None:
        try:
            processo = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            valores = [int(linha.strip()) for linha in processo.stdout.splitlines() if linha.strip()]
            return sum(valores)
        except (FileNotFoundError, ValueError, subprocess.SubprocessError):
            return None

    def _executar(self):
        while not self._parar.is_set():
            atual = self._ler()
            if atual is not None:
                self.pico_mib = atual if self.pico_mib is None else max(self.pico_mib, atual)
            self._parar.wait(self.intervalo)

    def iniciar(self):
        self._thread.start()

    def finalizar(self):
        self._parar.set()
        self._thread.join(timeout=10)


def _registro_evento(evento: dict, documento_id: str) -> dict:
    return {
        "documento_id": documento_id,
        "inicio": evento["inicio"],
        "fim": evento["fim"],
        "tipo": evento["tipo_previsto"],
        "score": evento["score_bruto_modelo"],
    }


def executar(args, detector=None, verificador=None) -> dict:
    manifesto, fold, docs = validar_entrada(args.dados, args.execucao)
    if args.saida.exists():
        raise ValueError("saída já existe; use uma pasta nova")
    if detector is None:
        detector = DetectorBIO.de_artefato(
            args.execucao, permitir_download=False, decodificador="viterbi",
            lote_janelas=args.lote_janelas_bio,
        )
    if verificador is None:
        if args.modelo_ollama:
            verificador = VerificadorOllama(
                modelo=args.modelo_ollama, endpoint=args.endpoint_ollama,
            )
        else:
            verificador = VerificadorLlama.de_modelo(
                args.modelo_llm,
                revisao=args.revisao_llm,
                permitir_download=args.permitir_download_llm,
                lote=args.lote_llm,
            )

    torch = None
    try:
        import torch as pacote_torch
        torch = pacote_torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    medidor_vram = _MedidorVRAM()
    medidor_vram.iniciar()
    inicio_tempo = time.perf_counter()
    verdade, regras, lora_bruto, fusao_final = [], [], [], []
    decisoes_llm = []
    por_decisao = {nome: [] for nome in ("CITACAO", "INCERTO", "NAO_CITACAO")}
    invalidas = fragmentos = documentos_invalidos = 0

    for documento_id, doc in docs.items():
        verdade.extend(_com_documento(span, documento_id) for span in doc["spans"])
        regras_doc = filtrar_distratores(extrair(doc["texto"]), doc["texto"])
        regras.extend(_com_documento(span, documento_id) for span in regras_doc)
        spans = detector.detectar(doc["texto"])
        lora_bruto.extend(_com_documento({
            "inicio": span.inicio, "fim": span.fim, "tipo": span.tipo,
            "score": span.score_bruto,
        }, documento_id) for span in spans)
        diagnostico = detector.ultimo_diagnostico
        invalidas += diagnostico.invalidas
        fragmentos += diagnostico.fragmentos
        documentos_invalidos += int(diagnostico.invalidas > 0)

        resultado = fusionar(
            regras_doc, spans, doc["texto"],
            limiares={"LEI": 0.0, "JUR": 0.0},
            filtro_distrator=e_distrator,
            verificador=verificador,
        )
        fusao_final.extend(_com_documento(span, documento_id) for span in resultado.candidatos)
        for evento in resultado.eventos:
            verificacao = evento.get("verificador")
            if verificacao is not None:
                por_decisao[verificacao["decisao"]].append(_registro_evento(evento, documento_id))
                decisoes_llm.append({
                    "documento_id": documento_id,
                    "inicio": evento["inicio"],
                    "fim": evento["fim"],
                    "tipo": evento["tipo_previsto"],
                    "trecho": doc["texto"][evento["inicio"]:evento["fim"]],
                    "score_lora": evento["score_bruto_modelo"],
                    **verificacao,
                })

    duracao = time.perf_counter() - inicio_tempo
    medidor_vram.finalizar()
    erros_regras = _detalhar_erros(verdade, regras, docs)
    exclusivos = [item for grupo in por_decisao.values() for item in grupo]
    aceitos = por_decisao["CITACAO"]
    metricas_exclusivos = avaliar_spans(erros_regras["falsos_negativos"], exclusivos)
    metricas_aceitos = avaliar_spans(erros_regras["falsos_negativos"], aceitos)
    metricas_fusao = _metricas(verdade, fusao_final, docs)
    pico_vram = None
    if torch is not None and torch.cuda.is_available():
        pico_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    resumo = {
        "protocolo": f"lora_alto_recall_llm_{PROMPT_VERSAO}",
        "fold": fold,
        "avaliacao_independente": False,
        "dataset_sha256": manifesto["dataset_sha256"],
        "prompt_sha256": PROMPT_SHA256,
        "prompt_versao": PROMPT_VERSAO,
        "politica": {
            "decodificador": "viterbi",
            "limiares_candidatos": {"LEI": 0.0, "JUR": 0.0},
            "entrada_llm": "somente_trecho",
            "few_shot": False,
            "aceitas": ["CITACAO"],
            "rejeitadas": ["INCERTO", "NAO_CITACAO"],
        },
        "n_documentos": len(docs),
        "n_entidades": len(verdade),
        "regras": _metricas(verdade, regras, docs),
        "lora_bruto": _metricas(verdade, lora_bruto, docs),
        "exclusivos_lora": {
            "quantidade": len(exclusivos),
            "contra_gold_nao_coberto_por_regras": metricas_exclusivos,
        },
        "verificador": {
            "contagens": {decisao: len(itens) for decisao, itens in por_decisao.items()},
            "por_decisao": {
                decisao: avaliar_spans(erros_regras["falsos_negativos"], itens)
                for decisao, itens in por_decisao.items()
            },
            "aceitos_contra_gold_nao_coberto_por_regras": metricas_aceitos,
        },
        "fusao_regras_llm": metricas_fusao,
        "recall_adicional_bruto_sobre_regras": (
            metricas_exclusivos["geral"]["tp"] / len(verdade) if verdade else None
        ),
        "recall_adicional_aceito_sobre_regras": (
            metricas_aceitos["geral"]["tp"] / len(verdade) if verdade else None
        ),
        "fp_realmente_acrescentados": metricas_aceitos["geral"]["fp"],
        "diagnostico_bio": {
            "sequencias_invalidas": invalidas,
            "spans_fragmentados": fragmentos,
            "documentos_com_bio_invalido": documentos_invalidos,
            "taxa_documentos_com_bio_invalido": documentos_invalidos / len(docs),
        },
        "recursos": {
            "tempo_total_segundos": duracao,
            "pico_vram_pipeline_mib": medidor_vram.pico_mib,
            "pico_vram_torch_processo_mib": pico_vram,
            "nota_vram": "pipeline soma processos visíveis no nvidia-smi; pode incluir outros usuários",
        },
    }
    detalhes = {
        "decisoes_llm": decisoes_llm,
        "regras": erros_regras,
        "lora_bruto": _detalhar_erros(verdade, lora_bruto, docs),
        "exclusivos_lora": _detalhar_erros(erros_regras["falsos_negativos"], exclusivos, docs),
        "aceitos_llm": _detalhar_erros(erros_regras["falsos_negativos"], aceitos, docs),
        "fusao": _detalhar_erros(verdade, fusao_final, docs),
    }
    args.saida.mkdir(parents=True, exist_ok=False)
    (args.saida / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.saida / "erros_locais.json").write_text(
        json.dumps(detalhes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, required=True)
    parser.add_argument("--execucao", type=Path, required=True)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--modelo-llm")
    grupo.add_argument("--modelo-ollama")
    parser.add_argument("--endpoint-ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--revisao-llm")
    parser.add_argument("--permitir-download-llm", action="store_true")
    parser.add_argument("--lote-llm", type=int, default=8)
    parser.add_argument("--lote-janelas-bio", type=int, default=8)
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    executar(args)
    print(args.saida / "resumo.json")


if __name__ == "__main__":
    main()
