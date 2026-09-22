"""Contratos entre extração, normalização, consulta e decisão."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Familia = Literal["lei", "sumula", "tema", "descritiva", "acordao"]
Classe = Literal["real", "inventada", "incompleta"]
Cobertura = Literal["indexada", "fora_base", "nao_consultada"]


@dataclass(frozen=True, slots=True)
class CitacaoNormalizada:
    familia: Familia
    inicio: int
    fim: int
    trecho: str
    normalizado: str
    tribunal: str | None = None
    classe: str | None = None
    numero: str | None = None
    uf: str | None = None
    origem: str | None = None
    ano: str | None = None
    artigo: str | None = None
    diploma: str | None = None
    vinculante: bool = False
    lei_numerada: bool = False
    identificador_suficiente: bool = False


@dataclass(frozen=True, slots=True)
class ResultadoResolucao:
    classe: Classe
    id_canonico: int | None
    caminho: str
    candidatos: tuple[int, ...] = ()
    cobertura: Cobertura = "indexada"
    motivo: str | None = None

    def __post_init__(self) -> None:
        if self.classe == "real" and self.id_canonico is None:
            raise ValueError("Citação real requer id_canonico")
        if self.classe != "real" and self.id_canonico is not None:
            raise ValueError("Apenas citação real pode ter id_canonico")
