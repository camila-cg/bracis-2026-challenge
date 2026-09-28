"""Métricas principais em spans, independentes de bibliotecas de ML."""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

from .fusao import iou


def avaliar_spans(verdade: list[dict], predicoes: list[dict], iou_minimo: float = 0.5) -> dict:
    usados: set[int] = set()
    pares: list[tuple[dict, dict, float]] = []
    for esperado in verdade:
        tipo = _tipo(esperado)
        opcoes = [
            (indice, previsto, iou(esperado["inicio"], esperado["fim"], previsto["inicio"], previsto["fim"]))
            for indice, previsto in enumerate(predicoes)
            if indice not in usados and _tipo(previsto) == tipo and _mesmo_documento(esperado, previsto)
        ]
        opcoes = [item for item in opcoes if item[2] >= iou_minimo]
        if opcoes:
            indice, previsto, valor = max(opcoes, key=lambda item: item[2])
            usados.add(indice)
            pares.append((esperado, previsto, valor))
    global_ = _prf(len(pares), len(predicoes) - len(pares), len(verdade) - len(pares))
    por_tipo = {}
    for tipo in ("LEI", "JUR"):
        v = [x for x in verdade if _tipo(x) == tipo]
        p = [x for x in predicoes if _tipo(x) == tipo]
        tp = sum(1 for esperado, _, _ in pares if _tipo(esperado) == tipo)
        por_tipo[tipo] = _prf(tp, len(p) - tp, len(v) - tp)
    return {"geral": global_, "por_tipo": por_tipo, "correspondencias": len(pares)}


def avaliar_fusao(verdade: list[dict], regras: list[dict], modelo: list[dict], fusao: list[dict]) -> dict:
    regras_m = avaliar_spans(verdade, regras)
    modelo_m = avaliar_spans(verdade, modelo)
    fusao_m = avaliar_spans(verdade, fusao)
    tp_regras = regras_m["geral"]["tp"]
    tp_fusao = fusao_m["geral"]["tp"]
    acrescentados = [p for p in fusao if p.get("evidencia_deteccao", {}).get("origem") == "modelo"]
    fp_acrescentados = max(0, fusao_m["geral"]["fp"] - regras_m["geral"]["fp"])
    return {
        "regras": regras_m,
        "modelo": modelo_m,
        "fusao": fusao_m,
        "recall_adicional_sobre_regras": (tp_fusao - tp_regras) / len(verdade) if verdade else 0.0,
        "falsos_positivos_acrescentados": fp_acrescentados,
        "taxa_fp_acrescentados": fp_acrescentados / len(acrescentados) if acrescentados else 0.0,
    }


def resumir_folds(resultados: list[dict]) -> dict:
    f1s = [item["fusao"]["geral"]["f1"] for item in resultados]
    return {"f1_medio": mean(f1s) if f1s else 0.0, "f1_desvio": pstdev(f1s) if len(f1s) > 1 else 0.0}


def analisar_erros(verdade_nao_casada: list[dict], pred_nao_casada: list[dict], limite: int = 20) -> dict:
    def chave(item: dict) -> str:
        trecho = item.get("trecho", "").lower().strip()
        return " ".join(trecho.split()[:4]) or _tipo(item)
    return {
        "falsos_negativos_frequentes": Counter(map(chave, verdade_nao_casada)).most_common(limite),
        "falsos_positivos_frequentes": Counter(map(chave, pred_nao_casada)).most_common(limite),
    }


def _prf(tp: int, fp: int, fn: int) -> dict:
    precisao = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precisao": precisao, "recall": recall, "f1": f1}


def _tipo(item: dict) -> str:
    tipo = str(item.get("tipo", "")).upper()
    return "LEI" if tipo == "LEI" else ("JUR" if tipo in {"JUR", "JURISPRUDENCIA", "JURISPRUDÊNCIA"} else tipo)


def _mesmo_documento(a: dict, b: dict) -> bool:
    """Evita casar offsets locais iguais que pertencem a documentos diferentes."""
    documento_a = a.get("documento_id")
    documento_b = b.get("documento_id")
    if documento_a is None and documento_b is None:
        return True
    return documento_a is not None and documento_a == documento_b
