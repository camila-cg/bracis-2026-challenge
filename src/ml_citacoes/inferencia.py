"""Carregamento local e inferência em janelas sobre o texto completo."""

from __future__ import annotations

import json
from pathlib import Path

from .bio import bio_para_spans, combinar_logits_janelas
from .contratos import CitacaoModelo


class DetectorBIO:
    """Adaptador opcional. Dependências pesadas são importadas apenas ao ativá-lo."""

    def __init__(self, tokenizer, modelo, *, max_length: int = 512, stride: int = 96,
                 temperatura: float = 1.0, decodificador: str = "viterbi",
                 lote_janelas: int = 8):
        if decodificador not in {"argmax", "viterbi"}:
            raise ValueError("decodificador deve ser argmax ou viterbi")
        if lote_janelas < 1:
            raise ValueError("lote_janelas deve ser positivo")
        self.tokenizer = tokenizer
        self.modelo = modelo
        self.max_length = max_length
        self.stride = stride
        self.temperatura = temperatura
        self.decodificador = decodificador
        self.lote_janelas = lote_janelas
        self.ultimo_diagnostico = None
        self.ultimo_diagnostico_bruto = None

    @classmethod
    def de_artefato(cls, pasta: Path, *, permitir_download: bool = False,
                   decodificador: str = "viterbi", lote_janelas: int = 8) -> "DetectorBIO":
        # Política de inferência explícita; não reescreve pesos/manifestos históricos.
        if decodificador not in {"argmax", "viterbi"}:
            raise ValueError("decodificador deve ser argmax ou viterbi")
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "inferência BIO requer torch e transformers; consulte requirements-ml.txt"
            ) from exc
        pasta = Path(pasta)
        manifesto_path = pasta / "manifesto.json"
        if manifesto_path.exists():
            manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
            checkpoint = manifesto["checkpoint"]
            revisao = manifesto.get("revisao_resolvida")
            if not revisao:
                raise ValueError("artefato sem revisão exata do checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(
                pasta / "tokenizer" if (pasta / "tokenizer").exists() else checkpoint,
                revision=None if (pasta / "tokenizer").exists() else revisao,
                local_files_only=not permitir_download,
                use_fast=True,
            )
            base = AutoModelForTokenClassification.from_pretrained(
                checkpoint,
                revision=revisao,
                local_files_only=not permitir_download,
                num_labels=5,
                id2label={0: "O", 1: "B-LEI", 2: "I-LEI", 3: "B-JUR", 4: "I-JUR"},
                label2id={"O": 0, "B-LEI": 1, "I-LEI": 2, "B-JUR": 3, "I-JUR": 4},
            )
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("artefato LoRA requer peft; consulte requirements-ml.txt") from exc
            modelo = PeftModel.from_pretrained(base, pasta / "adaptador")
            max_length = int(manifesto.get("max_length", 512))
            stride = int(manifesto.get("stride", 96))
            temperatura = float(manifesto.get("temperatura", 1.0))
        elif (pasta / "config.json").exists() and (pasta / "model.safetensors").exists():
            # Snapshot HF já mesclado: todos os pesos e o tokenizer ficam locais.
            tokenizer = AutoTokenizer.from_pretrained(pasta, local_files_only=True, use_fast=True)
            modelo = AutoModelForTokenClassification.from_pretrained(pasta, local_files_only=True)
            labels = {int(indice): rotulo for indice, rotulo in modelo.config.id2label.items()}
            esperados = {0: "O", 1: "B-LEI", 2: "I-LEI", 3: "B-JUR", 4: "I-JUR"}
            if labels != esperados:
                raise ValueError(f"rótulos incompatíveis no modelo mesclado: {labels}")
            max_length, stride, temperatura = 512, 96, 1.0
        else:
            raise ValueError(
                "artefato inválido: esperado LoRA com manifesto.json ou modelo mesclado "
                "com config.json e model.safetensors"
            )
        modelo.eval()
        if torch.cuda.is_available():
            modelo.to("cuda")
        return cls(
            tokenizer, modelo,
            max_length=max_length,
            stride=stride,
            temperatura=temperatura,
            decodificador=decodificador,
            lote_janelas=lote_janelas,
        )

    def detectar(self, texto: str) -> list[CitacaoModelo]:
        offsets, logits = self.obter_logits(texto)
        return self.decodificar(texto, offsets, logits)

    def obter_logits(self, texto: str):
        """Uma passagem local; permite ablações pareadas sem novo treinamento."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("inferência BIO requer torch") from exc
        lote = self.tokenizer(
            texto,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
        )
        offsets = lote.pop("offset_mapping").tolist()
        lote.pop("overflow_to_sample_mapping", None)
        dispositivo = next(self.modelo.parameters()).device
        logits = []
        total_janelas = len(offsets)
        with torch.inference_mode():
            for inicio in range(0, total_janelas, self.lote_janelas):
                fim = inicio + self.lote_janelas
                entradas = {
                    chave: valor[inicio:fim].to(dispositivo)
                    for chave, valor in lote.items()
                }
                logits.extend(self.modelo(**entradas).logits.detach().cpu().tolist())
        return offsets, logits

    def decodificar(self, texto: str, offsets, logits, *, decodificador: str | None = None):
        """Viterbi por padrão; argmax explícito preserva a comparação histórica."""
        decodificador = self.decodificador if decodificador is None else decodificador
        off_bruto, lab_bruto, score_bruto = combinar_logits_janelas(offsets, logits, decodificador="argmax")
        _, self.ultimo_diagnostico_bruto = bio_para_spans(lab_bruto, off_bruto, score_bruto)
        offsets_unicos, labels, scores = combinar_logits_janelas(offsets, logits, decodificador=decodificador)
        spans, diagnostico = bio_para_spans(labels, offsets_unicos, scores)
        calibrados = {}
        if self.temperatura != 1.0:
            logits_calibrados = [
                [[valor / self.temperatura for valor in token] for token in janela]
                for janela in logits
            ]
            off_cal, lab_cal, score_cal = combinar_logits_janelas(offsets, logits_calibrados, decodificador=decodificador)
            spans_cal, _ = bio_para_spans(lab_cal, off_cal, score_cal)
            calibrados = {(s.inicio, s.fim, s.tipo): s.score_bruto for s in spans_cal}
        self.ultimo_diagnostico = diagnostico
        return [
            CitacaoModelo(
                span.inicio, span.fim, span.tipo, span.score_bruto,
                calibrados.get((span.inicio, span.fim, span.tipo)),
                _identificar_ocr(texto[span.inicio:span.fim]),
            )
            for span in spans
        ]


def _identificar_ocr(trecho: str) -> tuple[str, ...]:
    transformacoes = []
    for indice, caractere in enumerate(trecho):
        if caractere in "lIOS" and (
            (indice and trecho[indice - 1].isdigit())
            or (indice + 1 < len(trecho) and trecho[indice + 1].isdigit())
        ):
            transformacoes.append(f"{caractere}@{indice}")
    return tuple(transformacoes)
