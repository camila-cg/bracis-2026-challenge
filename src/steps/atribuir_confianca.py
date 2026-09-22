"""ETAPA 6 — Confiança: caminho de decisão -> confiança."""

from __future__ import annotations

# Confiança por caminho de decisão. Chute inicial conservador; recalibrar
# por caminho com --avaliar (e com cuidado: é o dev set).
CONFIANCA = {
    "lei_real": 0.95,
    "lei_inventada": 0.90,
    "lei_sem_artigo": 0.60,
    "lei_sem_estatuto": 0.60,
    "sumula_real": 0.95,
    "sumula_inventada": 0.90,
    "sumula_sem_numero": 0.60,
    "sumula_sem_tribunal": 0.60,
    "tema": 0.85,
    "descritiva": 0.90,
    "acordao_indice": 0.97,
    "acordao_fts": 0.90,
    "acordao_zero": 0.85,
    "acordao_ambigua": 0.50,
}


def atribuir_confianca(caminho: str) -> float:
    return CONFIANCA[caminho]
