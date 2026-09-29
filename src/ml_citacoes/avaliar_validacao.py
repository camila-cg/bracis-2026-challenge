"""Avalia um fold, escolhe limiares somente na validação e mede o pipeline completo."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import eda_utils as u
from steps.buscar_na_base import Resolvedor
from steps.extrair_candidatos import extrair
from steps.filtrar_distratores import e_distrator, filtrar_distratores
from steps.normalizar_citacao import normalizar_citacao
from steps.rotear_citacao import rotear

from .contratos import CitacaoModelo
from .fusao import fusionar, iou
from .metricas import analisar_erros, avaliar_fusao, avaliar_spans


def avaliar(args) -> dict:
    documentos = {
        doc["documento_id"]: doc
        for doc in _ler_jsonl(args.dados / "documentos.jsonl")
    }
    manifesto = json.loads((args.execucao / "manifesto.json").read_text(encoding="utf-8"))
    ids_validacao = manifesto["documentos_validacao"]
    predicoes = json.loads((args.execucao / "predicoes_validacao.json").read_text(encoding="utf-8"))
    preparados = {}
    for documento_id in ids_validacao:
        documento = documentos[documento_id]
        regras = filtrar_distratores(extrair(documento["texto"]), documento["texto"])
        modelo = [_citacao_modelo(item) for item in predicoes.get(documento_id, [])]
        preparados[documento_id] = (documento, regras, modelo)

    grades = [0.90, 0.925, 0.95, 0.975, 0.99]
    regras_global = [
        _com_documento(r, documento_id)
        for documento_id, (_, regras, _) in preparados.items()
        for r in regras
    ]
    verdade_global = [
        _com_documento(s, documento_id)
        for documento_id, (doc, _, _) in preparados.items()
        for s in doc["spans"]
    ]
    precisao_regras = avaliar_spans(verdade_global, regras_global)["geral"]["precisao"]
    opcoes = []
    for limiar_lei in grades:
        for limiar_jur in grades:
            metrica, fusoes = _avaliar_limiares(preparados, {"LEI": limiar_lei, "JUR": limiar_jur})
            geral = metrica["fusao"]["geral"]
            if geral["precisao"] + 0.01 < precisao_regras:
                continue
            opcoes.append((geral["f1"], geral["precisao"], limiar_lei + limiar_jur, limiar_lei, limiar_jur, metrica, fusoes))
    if not opcoes:
        raise ValueError("nenhum limiar preservou a precisão das regras")
    _, _, _, limiar_lei, limiar_jur, metricas, fusoes = max(opcoes, key=lambda x: x[:3])
    por_nivel = {}
    for nivel in ("1", "2"):
        ids = [i for i in ids_validacao if f"_n{nivel}_" in i]
        verdade = [_com_documento(s, i) for i in ids for s in documentos[i]["spans"]]
        previsto = [_com_documento(s, i) for i in ids for s in fusoes[i]]
        por_nivel[nivel] = avaliar_spans(verdade, previsto)
    descritivas = [
        s for s in verdade_global
        if not any(ch.isdigit() for ch in s.get("trecho", ""))
    ]
    fusao_global = [
        _com_documento(s, documento_id)
        for documento_id, itens in fusoes.items()
        for s in itens
    ]
    metricas["descritivas"] = avaliar_spans(descritivas, fusao_global)
    metricas["por_nivel"] = por_nivel
    metricas["ocr"] = por_nivel["2"]
    metricas["limiares"] = {"LEI": limiar_lei, "JUR": limiar_jur}
    diagnostico = json.loads((args.execucao / "diagnostico_bio.json").read_text(encoding="utf-8"))
    total_pred = diagnostico.get("total_spans_preditos", sum(len(v) for v in predicoes.values()))
    metricas["taxa_sequencias_bio_invalidas"] = diagnostico.get("documentos_com_bio_invalido", 0) / max(1, diagnostico.get("total_documentos", len(ids_validacao)))
    metricas["taxa_spans_fragmentados"] = diagnostico["spans_fragmentados"] / max(1, total_pred)
    metricas["analise_erros"] = _erros(verdade_global, fusao_global)
    modelo_global = [
        _com_documento({"inicio": p.inicio, "fim": p.fim, "tipo": p.tipo, "score": p.score_bruto}, documento_id)
        for documento_id, (_, _, modelo) in preparados.items()
        for p in modelo
    ]
    metricas["erros_modelo"] = _detalhar_erros(verdade_global, modelo_global, documentos)
    if args.db:
        metricas["pipeline_completo_pos_base"] = _avaliar_pipeline_completo(preparados, fusoes, args.db)
    (args.execucao / "resultados_validacao.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metricas


def _avaliar_limiares(preparados, limiares):
    verdade, regras_todas, modelo_todo, fusao_toda = [], [], [], []
    fusoes = {}
    for documento_id, (documento, regras, modelo) in preparados.items():
        resultado = fusionar(regras, modelo, documento["texto"], limiares=limiares, filtro_distrator=e_distrator)
        fusoes[documento_id] = list(resultado.candidatos)
        verdade.extend(_com_documento(s, documento_id) for s in documento["spans"])
        regras_todas.extend(_com_documento(s, documento_id) for s in regras)
        modelo_todo.extend(_com_documento({"inicio": p.inicio, "fim": p.fim, "tipo": p.tipo}, documento_id) for p in modelo)
        fusao_toda.extend(_com_documento(s, documento_id) for s in resultado.candidatos)
    return avaliar_fusao(verdade, regras_todas, modelo_todo, fusao_toda), fusoes


def _avaliar_pipeline_completo(preparados, fusoes, db: Path) -> dict:
    con = sqlite3.connect(u._uri_sqlite(db.resolve()) + "?immutable=1", uri=True)
    resolvedor = Resolvedor(con)
    certos, casados = 0, 0
    for documento_id, (documento, _, _) in preparados.items():
        previstos = []
        for candidato in fusoes[documento_id]:
            resultado = resolvedor.resolver_citacao(normalizar_citacao(candidato, rotear(candidato)))
            previstos.append((candidato, resultado))
        usados = set()
        for esperado in documento["spans"]:
            opcoes = [
                (i, c, r, iou(esperado["inicio"], esperado["fim"], c["inicio"], c["fim"]))
                for i, (c, r) in enumerate(previstos)
                if i not in usados and c["tipo"] == esperado["tipo"]
            ]
            opcoes = [x for x in opcoes if x[3] >= 0.5]
            if not opcoes:
                continue
            indice, _, resultado, _ = max(opcoes, key=lambda x: x[3])
            usados.add(indice)
            casados += 1
            id_esperado = str(esperado.get("id_canonico") or "")
            id_previsto = str(resultado.id_canonico or "")
            if resultado.classe == esperado.get("classificacao") and id_previsto == id_esperado:
                certos += 1
    con.close()
    return {"pares_casados": casados, "classificacao_e_id_corretos": certos, "taxa_acerto": certos / casados if casados else 0.0}


def _erros(verdade, previsto):
    usados = set()
    fn = []
    for esperado in verdade:
        tipo_esperado = "LEI" if esperado["tipo"] == "lei" else "JUR"
        opcoes = [(i, p) for i, p in enumerate(previsto) if i not in usados and p.get("documento_id") == esperado.get("documento_id") and _tipo_item(p) == tipo_esperado and iou(esperado["inicio"], esperado["fim"], p["inicio"], p["fim"]) >= 0.5]
        if opcoes:
            usados.add(opcoes[0][0])
        else:
            fn.append(esperado)
    fp = [p for i, p in enumerate(previsto) if i not in usados]
    return analisar_erros(fn, fp)


def _detalhar_erros(verdade, previsto, documentos):
    usados = set()
    falsos_negativos = []
    for esperado in verdade:
        tipo_esperado = _tipo_item(esperado)
        opcoes = [
            (i, p, iou(esperado["inicio"], esperado["fim"], p["inicio"], p["fim"]))
            for i, p in enumerate(previsto)
            if i not in usados
            and p.get("documento_id") == esperado.get("documento_id")
            and _tipo_item(p) == tipo_esperado
            and iou(esperado["inicio"], esperado["fim"], p["inicio"], p["fim"]) >= 0.5
        ]
        if opcoes:
            indice, _, _ = max(opcoes, key=lambda x: x[2])
            usados.add(indice)
        else:
            falsos_negativos.append(_com_trecho(esperado, documentos))
    falsos_positivos = [
        _com_trecho(p, documentos)
        for i, p in enumerate(previsto)
        if i not in usados
    ]
    return {
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
    }


def _com_trecho(item, documentos):
    documento_id = item["documento_id"]
    texto = documentos[documento_id]["texto"]
    return {
        "documento_id": documento_id,
        "inicio": item["inicio"],
        "fim": item["fim"],
        "tipo": _tipo_item(item),
        "trecho": texto[item["inicio"]:item["fim"]],
        **({"score": item["score"]} if "score" in item else {}),
    }


def _com_documento(item, documento_id):
    return {**item, "documento_id": documento_id}


def _citacao_modelo(item):
    tipo = "LEI" if item["tipo"].upper() == "LEI" else "JUR"
    return CitacaoModelo(item["inicio"], item["fim"], tipo, item["score"], item["score"])


def _tipo_item(item):
    tipo = item.get("tipo", "").upper()
    return "LEI" if tipo == "LEI" else "JUR"


def _ler_jsonl(caminho):
    with caminho.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", type=Path, required=True)
    parser.add_argument("--execucao", type=Path, required=True)
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    print(json.dumps(avaliar(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
