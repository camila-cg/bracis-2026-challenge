"""Avaliação local, opcional (--avaliar): métrica oficial contra o goldenset."""

from __future__ import annotations

import csv
from pathlib import Path

import eda_utils as u
from steps.pipeline_utils import DADOS, carregar_modulo


def avaliar(submission_csv: Path, goldenset: Path) -> dict:
    import pandas as pd

    metrica = carregar_modulo(DADOS / "kaggle_metric.py", "kaggle_metric")
    submission = pd.read_csv(submission_csv, dtype=str, keep_default_na=False)

    por_documento: dict[str, list[str]] = {d: [] for d in submission["documento_id"]}
    with goldenset.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            ids = r["id_canonico"].strip() or "-"
            por_documento.setdefault(r["documento_id"], []).append(
                f"{r['inicio']},{r['fim']},{r['classificacao']},{ids}")
    solution = pd.DataFrame([
        {"documento_id": d, "nivel": u.nivel(d), "citacoes": "|".join(c) or "-"}
        for d, c in por_documento.items()
    ])
    return metrica.avaliar(solution, submission, row_id="documento_id")


def imprimir_avaliacao(resultado: dict) -> None:
    print(f"\nscore_final: {resultado['score_final']:.4f}")
    for nivel, r in resultado["niveis"].items():
        f1 = " · ".join(f"{c} {v:.3f}" for c, v in r["f1_por_classe"].items())
        print(f"  nível {nivel}: score {r['score']:.4f} · macroF1 {r['macro_f1']:.4f} "
              f"· τ {r['tau']:.3f} · bônus {r['b']:.4f}  ({f1})")
