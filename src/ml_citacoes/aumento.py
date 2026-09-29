"""Aumentos de OCR seguros, destinados exclusivamente ao split de treino."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformacaoOCR:
    posicao: int
    original: str
    substituto: str


# Confusões recuperáveis vistas no regulamento. Nenhuma troca dígito -> dígito.
CONFUSOES = {"0": ("O",), "1": ("l", "I"), "5": ("S",), "m": ("rn",)}


def aumentar_citacao_real(trecho: str, *, seed: int, probabilidade: float = 0.15) -> tuple[str, tuple[TransformacaoOCR, ...]]:
    rng = random.Random(seed)
    saida: list[str] = []
    transformacoes: list[TransformacaoOCR] = []
    for posicao, caractere in enumerate(trecho):
        opcoes = CONFUSOES.get(caractere)
        if caractere.isdigit() and not _digito_em_bloco_recuperavel(trecho, posicao):
            opcoes = None
        if opcoes and rng.random() < probabilidade:
            substituto = rng.choice(opcoes)
            if caractere.isdigit() and substituto.isdigit():
                raise AssertionError("aumento inválido: dígito não pode virar outro dígito")
            saida.append(substituto)
            transformacoes.append(TransformacaoOCR(posicao, caractere, substituto))
        else:
            saida.append(caractere)
    return "".join(saida), tuple(transformacoes)


def _digito_em_bloco_recuperavel(trecho: str, posicao: int) -> bool:
    """Espelha o contexto aceito por LETRA_EM_NUMERO no normalizador atual."""
    anterior_e_digito = posicao > 0 and trecho[posicao - 1].isdigit()
    seguinte_e_digito = posicao + 1 < len(trecho) and trecho[posicao + 1].isdigit()
    depois_de_ponto_numerico = posicao > 1 and trecho[posicao - 1] == "." and trecho[posicao - 2].isdigit()
    return anterior_e_digito or seguinte_e_digito or depois_de_ponto_numerico
