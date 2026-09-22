"""Extrai atributos semânticos sem alterar o span ou o trecho original."""

from __future__ import annotations

import re

import eda_utils as u
import indice_utils as ind
from configuracao import carregar_config
from modelos import CitacaoNormalizada, Familia
from steps.normalizar_numeros import normalizar
from steps.pipeline_utils import BLOCO_NUMERICO, corrigir_ocr, tem_identificador
from steps.rotear_citacao import E_SUMULA, E_TEMA

_TRIBUNAL = re.compile(
    r"\b(?:" + "|".join(map(re.escape, carregar_config("tribunais")["siglas"])) + r")\b", re.I
)
_UF = ind.UF_CITADA
_ARTIGO = re.compile(r"\bart(?:igo)?\.?\s*(\d[\d.]*)", re.I)
_LEI_NUMERADA = re.compile(r"\blei\s*(?:n\s*[º°o.]?\s*)?\d[\d.]*(?:/\d{2,4})?", re.I)
_ANO = re.compile(r"\b(?:19|20)\d{2}\b")
_VINCULANTE = re.compile(r"vincul", re.I)
_APELIDOS = carregar_config("diplomas")["apelidos_estatuto"]


def _numero_apos(trecho: str, palavra: re.Pattern) -> str | None:
    marcador = palavra.search(trecho)
    resto = corrigir_ocr(trecho[marcador.end():] if marcador else trecho)
    m = BLOCO_NUMERICO.search(resto)
    if not m:
        return None
    return re.sub(r"\D", "", m.group()).lstrip("0") or None


def _diploma(trecho: str) -> str | None:
    texto = re.sub(r"\s+", " ", trecho.lower())
    for nome, apelidos in _APELIDOS.items():
        if any(apelido in texto for apelido in apelidos):
            return nome
    return None


def normalizar_citacao(candidato: dict, familia: Familia, *, ja_normalizado: bool = False) -> CitacaoNormalizada:
    trecho = candidato["trecho"]
    texto_normalizado = trecho if ja_normalizado else normalizar(trecho, familia)
    tribunal = _TRIBUNAL.search(u.corrigir_ocr_em_numeros(trecho))
    uf = _UF.search(trecho)
    ano = _ANO.search(trecho)
    artigo = _ARTIGO.search(trecho)
    numero = None
    if familia == "sumula":
        numero = _numero_apos(trecho, E_SUMULA)
    elif familia == "tema":
        numero = _numero_apos(trecho, E_TEMA)
    elif familia == "acordao":
        formas = u.variantes_numero(texto_normalizado)
        if formas:
            numero = ind.normalizar_numero(formas[0]) if u.CNJ.search(formas[0]) else re.sub(r"\D", "", formas[0])
    return CitacaoNormalizada(
        familia=familia,
        inicio=candidato["inicio"],
        fim=candidato["fim"],
        trecho=trecho,
        normalizado=texto_normalizado,
        tribunal=tribunal.group().upper() if tribunal else None,
        classe=ind._classe_citada(trecho) if familia == "acordao" else None,
        numero=numero,
        uf=uf.group(1).upper() if uf else None,
        ano=ano.group() if ano else None,
        artigo=artigo.group(1).replace(".", "") if artigo else None,
        diploma=_diploma(trecho) if familia == "lei" else None,
        vinculante=bool(_VINCULANTE.search(trecho)) if familia == "sumula" else False,
        lei_numerada=bool(_LEI_NUMERADA.search(trecho)) if familia == "lei" else False,
        identificador_suficiente=tem_identificador(trecho),
    )
