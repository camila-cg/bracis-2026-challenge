"""Vocabulários e parâmetros versionados em YAML 1.2 (subconjunto JSON).

O formato restrito evita uma dependência de parser e permite validar tipos na carga.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PASTA_CONFIG = Path(__file__).resolve().parents[1] / "config"
ARQUIVOS = frozenset({"tribunais", "classes_processuais", "diplomas", "regras_ocr", "pipeline"})


@lru_cache(maxsize=None)
def carregar_config(nome: str) -> dict:
    if nome not in ARQUIVOS:
        raise ValueError(f"Configuração desconhecida: {nome}")
    caminho = PASTA_CONFIG / f"{nome}.yaml"
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Erro ao carregar {caminho}: {exc}") from exc
    if not isinstance(dados, dict):
        raise ValueError(f"{caminho}: esperado objeto de configuração")
    return dados
