"""ETAPA 6 — Confiança: caminho de decisão -> confiança."""

from __future__ import annotations

from configuracao import carregar_config
from modelos import Fonte

_CONFIG = carregar_config("pipeline")

# Confiança por caminho de decisão. Chute inicial conservador; recalibrar
# por caminho com --avaliar (e com cuidado: é o dev set).
CONFIANCA = _CONFIG["confianca"]

# Desconto aplicado quando a citação veio de uma fonte além das regras fixas
# (ex. o modelo generalizável). 0.0 é proposital: esse extrator ainda não
# existe/não foi treinado, então não há dado empírico pra calibrar o desconto
# — o hook fica pronto e testado, sem mudar nada até haver essa medição.
DESCONTO_FONTE_MODELO = _CONFIG.get("desconto_fonte_modelo", 0.0)


def atribuir_confianca(caminho: str, fonte: Fonte = "regra") -> float:
    base = CONFIANCA[caminho]
    if fonte == "regra":
        return base
    # Um desconto negativo viraria bônus (1 - desconto > 1); a métrica do
    # desafio rejeita a submissão inteira se algum valor sair de [0, 1], então
    # o resultado é sempre travado nesse intervalo, não importa o sinal do
    # desconto configurado.
    return max(0.0, min(1.0, base * (1.0 - DESCONTO_FONTE_MODELO)))
