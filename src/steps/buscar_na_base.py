"""ETAPA 5 — Resolução: (rota, trecho) -> classe + id_canonico + caminho."""

from __future__ import annotations

import re
import sqlite3

import eda_utils as u
import indice_utils as ind
from steps.pipeline_utils import BLOCO_NUMERICO, corrigir_ocr, tem_identificador
from steps.rotear_citacao import E_SUMULA, ROTA_ACORDAO, ROTA_DESCRITIVA, ROTA_LEI, ROTA_SUMULA, ROTA_TEMA

E_VINCULANTE = re.compile(r"vincul", re.I)
TRIBUNAL = re.compile(r"\b(ST[FJMG]|TS[TE])\b")
LEI_NUMERADA = re.compile(r"\blei\s*(?:n\s*[º°o.]?\s*)?\d[\d.]*(?:/\d{2,4})?", re.I)

# Estatuto citado -> nome usado em indice_utils.ESTATUTO_POR_ID.
APELIDOS_ESTATUTO = {
    "Constituição": ["constitui", "cf/88", "carta magna", "cf/1988"],
    "CLT": ["clt", "consolidação das leis do trabalho", "consolidacao das leis do trabalho", "5.452"],
    "Código de Processo Civil": ["cpc", "processo civil", "13.105"],
    "Código de Processo Penal": ["cpp", "processo penal", "3.689"],
    "Código Penal Militar": ["cpm", "penal militar", "1.001"],
    "Código Civil": ["código civil", "codigo civil", "10.406"],
    "Código de Defesa do Consumidor": ["cdc", "defesa do consumidor", "8.078"],
    "Código Eleitoral": ["código eleitoral", "codigo eleitoral", "4.737"],
    "Lei Complementar nº 64/1990": ["complementar nº 64", "complementar n. 64", "complementar 64",
                                    "lc 64", "lc nº 64", "64/1990", "64/90"],
}


def estatuto_citado(trecho: str) -> str | None:
    t = re.sub(r"\s+", " ", trecho.lower())
    for nome, chaves in APELIDOS_ESTATUTO.items():
        if any(k in t for k in chaves):
            return nome
    return None


def numero_apos(trecho: str, palavra: re.Pattern) -> str | None:
    """Número que segue a palavra-chave (súmula, tema), com OCR corrigido.

    Procurar depois da palavra evita pegar o "5" de "5úmula".
    """
    m = palavra.search(trecho)
    resto = corrigir_ocr(trecho[m.end():] if m else trecho)
    bloco = BLOCO_NUMERICO.search(resto)
    if not bloco:
        return None
    return re.sub(r"\D", "", bloco.group(0)).lstrip("0") or None


class Resolvedor:
    """Índices da base canônica e uma regra de resolução por rota."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.acordaos, _ = ind.construir_indice_acordaos(con)
        self.normativos, _ = ind.construir_indice_normativos(con)

    def lei(self, trecho: str) -> tuple[str, int | None, str]:
        m = re.search(r"art(?:igo)?\.?\s*(\d[\d.]*)", trecho, re.I)
        if not m:
            return "incompleta", None, "lei_sem_artigo"
        artigo = m.group(1).replace(".", "")
        estatuto = estatuto_citado(trecho)
        if not estatuto and not LEI_NUMERADA.search(trecho):
            return "incompleta", None, "lei_sem_estatuto"
        alvo = self.normativos.get(f"lei|{estatuto}|{artigo}") if estatuto else None
        if alvo:
            return "real", alvo["id"], "lei_real"
        return "inventada", None, "lei_inventada"

    def sumula(self, trecho: str) -> tuple[str, int | None, str]:
        numero = numero_apos(trecho, E_SUMULA)
        if not numero:
            return "incompleta", None, "sumula_sem_numero"
        if E_VINCULANTE.search(trecho):
            chave = f"sumula|STF|VINCULANTE-{numero}"
        else:
            m = TRIBUNAL.search(u.corrigir_ocr_em_numeros(trecho).upper())
            if not m:
                return "incompleta", None, "sumula_sem_tribunal"
            chave = f"sumula|{m.group(1)}|{numero}"
        alvo = self.normativos.get(chave)
        if alvo:
            return "real", alvo["id"], "sumula_real"
        return "inventada", None, "sumula_inventada"

    def tema(self, trecho: str) -> tuple[str, int | None, str]:
        # Não há tema de repercussão geral na base: com número, não resolve.
        return ("inventada" if tem_identificador(trecho) else "incompleta"), None, "tema"

    def descritiva(self, trecho: str) -> tuple[str, int | None, str]:
        # Buscável, mas sem número não há como desempatar.
        return "incompleta", None, "descritiva"

    def acordao(self, trecho: str) -> tuple[str, int | None, str]:
        """Espera o trecho já normalizado (etapa 4)."""
        candidatos = ind.resolver(trecho, self.acordaos)
        caminho = "acordao_indice"
        if not candidatos:
            candidatos = set(u.resolver_por_numero(trecho, con=self.con))
            candidatos = ind.filtrar_candidatos(trecho, candidatos, self.acordaos, self.con)
            caminho = "acordao_fts"
        if len(candidatos) == 1:
            return "real", next(iter(candidatos)), caminho
        if not candidatos:
            return "inventada", None, "acordao_zero"
        return "incompleta", None, "acordao_ambigua"


def resolver(rota: str, trecho: str, resolvedor: Resolvedor) -> tuple[str, int | None, str]:
    regra = {
        ROTA_LEI: resolvedor.lei,
        ROTA_SUMULA: resolvedor.sumula,
        ROTA_TEMA: resolvedor.tema,
        ROTA_DESCRITIVA: resolvedor.descritiva,
        ROTA_ACORDAO: resolvedor.acordao,
    }[rota]
    return regra(trecho)
