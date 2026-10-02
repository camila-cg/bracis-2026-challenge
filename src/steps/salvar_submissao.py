"""ETAPA 7 — Saída: JSONs por documento -> submission.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from steps.pipeline_utils import DADOS, carregar_modulo


def gravar_saida(
    documentos: list[dict],
    saida: Path,
    arquivo_csv: Path | None = None,
) -> Path:
    pasta_json = saida / "json"
    pasta_json.mkdir(parents=True, exist_ok=True)
    for doc in documentos:
        (pasta_json / f"{doc['documento_id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(doc["citacoes"]) for doc in documentos)
    print(f"{len(documentos)} documentos, {total} citações -> {pasta_json}")

    conversor = carregar_modulo(DADOS / "json_to_submission.py", "json_to_submission")
    submission_csv = arquivo_csv or (saida / "submission.csv")
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    with submission_csv.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["documento_id", "citacoes"])
        writer.writerows((doc["documento_id"], conversor.encode(doc)) for doc in documentos)
    print(f"{submission_csv}: {len(documentos)} documentos.")
    return submission_csv
