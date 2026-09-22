"""ETAPA 3 — Roteamento: candidato -> rota."""

from __future__ import annotations

import re

from steps.pipeline_utils import tem_identificador

ROTA_LEI = "lei"
ROTA_SUMULA = "sumula"
ROTA_TEMA = "tema"
ROTA_DESCRITIVA = "descritiva"  # tribunal/classe + ano + relator, sem número
ROTA_ACORDAO = "acordao"

E_SUMULA = re.compile(r"\b(?:[s5$][uúü][mn]|verbete)", re.I)
E_TEMA = re.compile(r"\bt[eê]m[aã]\b", re.I)


def rotear(candidato: dict) -> str:
    trecho = candidato["trecho"]
    if candidato["tipo"] == "lei":
        return ROTA_LEI
    if E_SUMULA.search(trecho):
        return ROTA_SUMULA
    if E_TEMA.search(trecho):
        return ROTA_TEMA
    if not tem_identificador(trecho):
        return ROTA_DESCRITIVA
    return ROTA_ACORDAO
