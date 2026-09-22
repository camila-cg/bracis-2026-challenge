"""Peças usadas por mais de uma etapa: caminhos, carga de módulo, OCR e blocos numéricos."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import eda_utils as u

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data"


def carregar_modulo(caminho: Path, nome: str):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# "de 2023", "em 2021": ano de julgamento, não identificador. Sem isso a
# incompleta descritiva vira busca pelo número "2.023".
ANO_SOLTO = re.compile(r"\b(?:de|em)\s+(?:19|20)\d{2}\b", re.I)

# Letra no lugar de dígito, só quando colada a um dígito ("21737l8", "6G.838",
# "170076O"). Exigir o dígito vizinho é o que impede o "S" de "- SP" de virar 5.
OCR_LETRA = "OoSslIigqBZzGbTD"
LETRA_EM_NUMERO = re.compile(rf"(?<=\d)[{OCR_LETRA}]|(?<=\d\.)[{OCR_LETRA}]|[{OCR_LETRA}](?=\d)")

# Bloco numérico: dígitos separados por qualquer mistura de ponto, espaço,
# quebra de linha e hífen ("33.-\n474", "1.741. 784", "7220273--\n23.2018...").
# Um dígito nunca vira outro (garantia do desafio), então juntar os dígitos
# do bloco reconstrói o identificador; o separador é só ruído.
BLOCO_NUMERICO = re.compile(r"\d(?:[.\s\-–]*\d)*")


def corrigir_ocr(trecho: str) -> str:
    # Repete até estabilizar: em "1OO" o segundo O só fica colado a um dígito
    # depois que o primeiro vira 0.
    anterior = None
    while anterior != trecho:
        anterior = trecho
        trecho = LETRA_EM_NUMERO.sub(lambda m: u.OCR_PARA_DIGITO[m.group(0)], trecho)
    return trecho


def tem_identificador(trecho: str) -> bool:
    limpo = ANO_SOLTO.sub(" ", corrigir_ocr(trecho))
    return any(len(re.sub(r"\D", "", m.group(0))) >= 3 for m in BLOCO_NUMERICO.finditer(limpo))
