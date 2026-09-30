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
DIGITOS_CNJ_MAX = 20  # sete dígitos sequenciais + 13 dígitos estruturados


def canonizar_bloco(digitos: str) -> str:
    if DIGITOS_CNJ_MIN <= len(digitos) <= DIGITOS_CNJ_MAX:
        seq, fim = digitos[:-13], digitos[-13:]
        return f"{seq}-{fim[:2]}.{fim[2:6]}.{fim[6]}.{fim[7:9]}.{fim[9:]}"
    return u.agrupar_milhares(digitos)


_CAUDA_CNJ_ISOLADA = re.compile(r"\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")


def _canonizar_match(m: re.Match) -> str:
    bruto = m.group(0)
    if _CAUDA_CNJ_ISOLADA.fullmatch(bruto):
        # A cauda de um CNJ sozinha é um CNJ partido (separador estranho antes
        # do DV). Reagrupá-la em milhares apagaria a estrutura e deixaria só o
        # sequencial como "número"; preservada, a validação estrita a rejeita.
        return bruto
    digitos = re.sub(r"\D", "", bruto)
    if len(digitos) >= DIGITOS_CNJ_MIN and re.search(r"[-–]", bruto):
        sequencial, fim = re.split(r"[-–]", bruto, maxsplit=1)
        digitos_sequencial = re.sub(r"\D", "", sequencial)
        digitos_fim = re.sub(r"\D", "", fim)
        if not (1 <= len(digitos_sequencial) <= 7 and len(digitos_fim) == 13):
            # Preserva o candidato inválido para que a validação estrita o
            # rejeite inteiro; reagrupá-lo esconderia a estrutura CNJ.
            return bruto
    return canonizar_bloco(digitos)


def normalizar_trecho(trecho: str) -> str:
    """Trecho com cada bloco numérico reescrito na forma canônica do índice."""
    # Um travessão com espaços antes da cauda do CNJ ("1036 — 78.2013...") é o
    # hífen do CNJ; sem isso o bloco numérico acaba em "1036".
    return BLOCO_NUMERICO.sub(_canonizar_match, u.normalizar_traco_de_cnj(corrigir_ocr(trecho)))


def normalizar(trecho: str, rota: str) -> str:
    """Só acórdão é buscado pelo número inteiro. Lei e súmula leem o número
    depois da palavra-chave, já tolerando OCR (etapa 5); tema e descritiva
    não consultam a base."""
    if rota == ROTA_ACORDAO:
        return normalizar_trecho(trecho)
    return trecho
