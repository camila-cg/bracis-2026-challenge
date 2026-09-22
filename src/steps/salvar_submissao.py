"""ETAPA 7 — Saída: JSONs por documento -> submission.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from steps.pipeline_utils import DADOS, carregar_modulo


def gravar_saida(documentos: list[dict], saida: Path) -> Path:
    pasta_json = saida / "json"
    pasta_json.mkdir(parents=True, exist_ok=True)
    for doc in documentos:
        (pasta_json / f"{doc['documento_id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(doc["citacoes"]) for doc in documentos)
    print(f"{len(documentos)} documentos, {total} citações -> {pasta_json}")

    conversor = carregar_modulo(DADOS / "json_to_submission.py", "json_to_submission")
    submission_csv = saida / "submission.csv"
    sys.argv = ["json_to_submission.py", str(pasta_json), str(submission_csv)]
    conversor.main()
    return submission_csv
