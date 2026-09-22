"""ETAPA 4 — Normalização: trecho -> trecho na forma do índice."""

from __future__ import annotations

import re

import eda_utils as u
from configuracao import carregar_config
from steps.pipeline_utils import BLOCO_NUMERICO, corrigir_ocr
from steps.rotear_citacao import ROTA_ACORDAO

# CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO — os 13 dígitos finais têm estrutura fixa,
# o sequencial às vezes vem sem os zeros à esquerda (TST/TSE).
DIGITOS_CNJ_MIN = carregar_config("pipeline")["digitos_cnj_min"]


def canonizar_bloco(digitos: str) -> str:
    if len(digitos) >= DIGITOS_CNJ_MIN:
        seq, fim = digitos[:-13], digitos[-13:]
        return f"{seq}-{fim[:2]}.{fim[2:6]}.{fim[6]}.{fim[7:9]}.{fim[9:]}"
    return u.agrupar_milhares(digitos)


def normalizar_trecho(trecho: str) -> str:
    """Trecho com cada bloco numérico reescrito na forma canônica do índice."""
    return BLOCO_NUMERICO.sub(
        lambda m: canonizar_bloco(re.sub(r"\D", "", m.group(0))), corrigir_ocr(trecho))


def normalizar(trecho: str, rota: str) -> str:
    """Só acórdão é buscado pelo número inteiro. Lei e súmula leem o número
    depois da palavra-chave, já tolerando OCR (etapa 5); tema e descritiva
    não consultam a base."""
    if rota == ROTA_ACORDAO:
        return normalizar_trecho(trecho)
    return trecho
