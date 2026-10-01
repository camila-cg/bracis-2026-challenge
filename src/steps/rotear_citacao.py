"""ETAPA 3 — Roteamento: candidato -> rota."""

from __future__ import annotations

import re
import unicodedata

from steps.pipeline_utils import tem_identificador

ROTA_LEI = "lei"
ROTA_SUMULA = "sumula"
ROTA_TEMA = "tema"
ROTA_DESCRITIVA = "descritiva"  # tribunal/classe + ano + relator, sem número
ROTA_ACORDAO = "acordao"

E_SUMULA = re.compile(
    r"\b(?:[s5$][uúü][mn]|verbete|enunciado|SV(?=\s+\d))",
    re.I,
)
E_TEMA = re.compile(r"\bt[eê]m[aã]\b", re.I)


def rotear(candidato: dict) -> str:
    # Regexes de rota devem enxergar NFD e NFC como a mesma superfície. Essa
    # composição é só semântica; os offsets continuam ligados ao texto bruto.
    trecho = unicodedata.normalize("NFC", candidato["trecho"])
    if candidato["tipo"] == "lei":
        return ROTA_LEI
    if E_SUMULA.search(trecho):
        return ROTA_SUMULA
    if E_TEMA.search(trecho):
        return ROTA_TEMA
    # O extrator já reconheceu o formato processual curto com UF. Reutilizar
    # esse sinal evita ampliar ``tem_identificador`` para qualquer número de
    # 1–2 dígitos em prosa.
    if any(
        registro.get("tipo") == "numero_processual"
        for registro in candidato.get("registros", [])
    ):
        return ROTA_ACORDAO
    if not tem_identificador(trecho):
        return ROTA_DESCRITIVA
    return ROTA_ACORDAO
