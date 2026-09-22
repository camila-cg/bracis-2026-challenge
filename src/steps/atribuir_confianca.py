"""ETAPA 6 — Confiança: caminho de decisão -> confiança."""

from __future__ import annotations

from configuracao import carregar_config

# Confiança por caminho de decisão. Chute inicial conservador; recalibrar
# por caminho com --avaliar (e com cuidado: é o dev set).
CONFIANCA = carregar_config("pipeline")["confianca"]


def atribuir_confianca(caminho: str) -> float:
    return CONFIANCA[caminho]
