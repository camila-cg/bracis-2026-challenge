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
VALOR_DA_CAUSA = re.compile(
    r"(?:valor\s+(?:atribu[ií]do\s+)?(?:da|à)\s+causa"
    r"|d[aá]-se\s+à\s+causa\s+o\s+valor)\s*(?::|de)?\s*"
    r"(?:R\$\s*)?(?P<quantia>\d{1,3}(?:\.\d{3})+(?:,\d{2})?|\d+(?:,\d{2})?)\b",
    re.I,
)


def _e_valor_da_causa(candidato: dict, texto: str) -> bool:
    inicio, fim = candidato["inicio"], candidato["fim"]
    if not (0 <= inicio < fim <= len(texto)):
        return False
    linha_inicio = texto.rfind("\n", 0, inicio) + 1
    linha_fim = texto.find("\n", fim)
    if linha_fim < 0:
        linha_fim = len(texto)
    linha = texto[linha_inicio:linha_fim]
    for campo in VALOR_DA_CAUSA.finditer(linha):
        # O candidato deve terminar no campo. Uma citação depois da quantia,
        # mesmo na mesma linha, não é um valor da causa.
        quantia_inicio = linha_inicio + campo.start("quantia")
        campo_fim = linha_inicio + campo.end()
        if inicio < campo_fim and fim > quantia_inicio and fim <= campo_fim:
            return True
    return False


def e_distrator(candidato: dict, texto: str | None = None) -> bool:
    if DISTRATOR.search(candidato["trecho"]):
        return True
    if texto is not None and _e_valor_da_causa(candidato, texto):
        return True
    return candidato["inicio"] < LIMITE_CABECALHO and bool(ABRE_COM_AUTOS.match(candidato["trecho"]))


def filtrar_distratores(candidatos: list[dict], texto: str | None = None) -> list[dict]:
    return [c for c in candidatos if not e_distrator(c, texto)]
