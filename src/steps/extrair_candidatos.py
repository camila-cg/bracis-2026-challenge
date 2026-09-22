"""ETAPA 1 — Extração: texto -> candidatos."""

from __future__ import annotations

import detectar_citacoes as dc
from configuracao import carregar_config

# Pontuação mínima do candidato no extrator. 4 é o piso: recupera todas as
# citações do dev set, e o que entra de lixo é distrator, tratado na etapa 2.
PONTUACAO_MINIMA = carregar_config("pipeline")["pontuacao_minima"]


def extrair(texto: str) -> list[dict]:
    """Candidatos com inicio, fim, trecho, tipo e pontuação acima do piso."""
    tokens = dc.criar_tokens(texto)
    registros = dc.detectar_marcadores_e_indicios(texto, tokens)
    grupos = dc.agrupar_registros(texto, registros)
    candidatos = dc.formar_candidatos(texto, tokens, grupos)
    return [c for c in candidatos if c["pontuacao"] >= PONTUACAO_MINIMA]
