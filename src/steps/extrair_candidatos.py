"""ETAPA 1 — Extração: texto -> candidatos."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import detectar_citacoes as dc
from configuracao import carregar_config

# Pontuação mínima do candidato no extrator. 4 é o piso: recupera todas as
# citações do dev set, e o que entra de lixo é distrator, tratado na etapa 2.
PONTUACAO_MINIMA = carregar_config("pipeline")["pontuacao_minima"]


def extrair(texto: str) -> list[dict]:
    """Candidatos com inicio, fim, trecho, tipo, fonte e pontuação acima do piso."""
    tokens = dc.criar_tokens(texto)
    registros = dc.detectar_marcadores_e_indicios(texto, tokens)
    grupos = dc.agrupar_registros(texto, registros)
    candidatos = dc.formar_candidatos(texto, tokens, grupos)
    # "fonte" identifica se o candidato veio das regras fixas (aqui) ou de um
    # extrator alternativo (ex. o modelo generalizável) — usado por
    # atribuir_confianca para diferenciar a confiança entre as duas origens.
    return [{**c, "fonte": "regra"} for c in candidatos if c["pontuacao"] >= PONTUACAO_MINIMA]


def gravar_debug_extracao(linhas: list[dict], pasta: Path, momento: datetime | None = None) -> Path:
    """CSV datado (um por execução, uma linha por documento) para depurar a extração.

    Colunas: documento_id, n_caracteres, texto, n_candidatos e candidatos (JSON
    com inicio, fim, trecho, tipo, pontuacao... de cada candidato extraído).
    """
    momento = momento or datetime.now()
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"extracao_{momento:%Y-%m-%d_%H-%M-%S}.csv"
    colunas = ["documento_id", "n_caracteres", "texto", "n_candidatos", "candidatos"]
    with caminho.open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=colunas)
        escritor.writeheader()
        for linha in linhas:
            escritor.writerow({
                "documento_id": linha["documento_id"],
                "n_caracteres": len(linha["texto"]),
                "texto": linha["texto"],
                "n_candidatos": len(linha["candidatos"]),
                "candidatos": json.dumps(linha["candidatos"], ensure_ascii=False, default=str),
            })
    return caminho
