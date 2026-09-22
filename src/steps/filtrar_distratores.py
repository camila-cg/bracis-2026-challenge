"""ETAPA 2 — Filtro de distratores: candidatos -> candidatos."""

from __future__ import annotations

import re

# O número dos autos do próprio documento fica no cabeçalho. Citação de outro
# processo em CNJ também pode vir como "processo nº TST-...", então o que
# separa os dois é a posição, não a palavra.
# Os rótulos das partes e do relator também são cabeçalho ("Reclamada: ...
# Relator: Desembargador ..."): o extrator pega o bloco pelo "Relator" e pelos
# nomes próprios.
LIMITE_CABECALHO = 400
ABRE_COM_AUTOS = re.compile(
    r"\s*(?:processo|autos|reclamad[ao]|reclamante|agravad[ao]|agravante"
    r"|recorrid[ao]|recorrente|relator[a]?)\b|\s*\d",
    re.I,
)
DISTRATOR = re.compile(r"R\$|\bfls?\.\s*\d|\bOAB\b|\bprotocolo\b", re.I)


def e_distrator(candidato: dict) -> bool:
    if DISTRATOR.search(candidato["trecho"]):
        return True
    return candidato["inicio"] < LIMITE_CABECALHO and bool(ABRE_COM_AUTOS.match(candidato["trecho"]))


def filtrar_distratores(candidatos: list[dict]) -> list[dict]:
    return [c for c in candidatos if not e_distrator(c)]
