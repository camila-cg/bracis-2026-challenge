"""Contratos compartilhados pelo token classifier BIO e pela fusão com regras.

Define a ordem fixa dos cinco rótulos usados pela cabeça de classificação, as conversões entre rótulos e IDs e as estruturas que transportam spans previstos e evidências de concordância com o detector por regras.

Este módulo não executa tokenização, inferência, calibração, fusão ou consulta à base canônica. Os scores armazenados são evidências do modelo e não devem ser
interpretados como a confiança final ou como a classificação jurídica da citação.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

LABELS = ("O", "B-LEI", "I-LEI", "B-JUR", "I-JUR")
LABEL2ID = {rotulo: indice for indice, rotulo in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
TipoBIO = Literal["LEI", "JUR"]
DecisaoVerificador = Literal["CITACAO", "INCERTO", "NAO_CITACAO"]


@dataclass(frozen=True, slots=True)
class CitacaoModelo:
    inicio: int
    fim: int
    tipo: TipoBIO
    score_bruto: float
    score_calibrado: float | None = None
    transformacoes_ocr: tuple[str, ...] = ()

    @property
    def tipo_pipeline(self) -> str:
        return "lei" if self.tipo == "LEI" else "jurisprudencia"


@dataclass(frozen=True, slots=True)
class EvidenciaDeteccao:
    origem: Literal["regra", "modelo", "ambos"]
    tipo_previsto: str | None = None
    score_bruto_modelo: float | None = None
    score_calibrado: float | None = None
    iou_regra: float | None = None
    relacao: Literal["concordancia", "discordancia_tipo", "somente_regra", "somente_modelo"] = "somente_regra"
    transformacoes_ocr: tuple[str, ...] = ()

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResultadoVerificacao:
    """Decisão textual sobre um span já delimitado pelo detector BIO."""

    decisao: DecisaoVerificador
    confianca: float
    probabilidades: dict[str, float]
    modelo: str
    prompt_sha256: str

    @property
    def aceita(self) -> bool:
        return self.decisao == "CITACAO"

    def como_dict(self) -> dict:
        return asdict(self)
