"""Seleciona pelos cinco folds sem deixar as regras mascararem um LoRA colapsado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def selecionar(legal: list[Path], bertimbau: list[Path]) -> dict:
    if len(legal) != 5 or len(bertimbau) != 5:
        raise ValueError("a comparação exige exatamente cinco folds de cada modelo")
    legal_m = [_manifesto(p) for p in legal]
    bert_m = [_manifesto(p) for p in bertimbau]
    for indice, (a, b) in enumerate(zip(legal_m, bert_m)):
        if a["fold"] != b["fold"] or a["seed"] != b["seed"] or a["documentos_validacao"] != b["documentos_validacao"] or a["dataset_sha256"] != b["dataset_sha256"]:
            raise ValueError(f"modelos não usaram o mesmo split/seed/dataset no par {indice}")
    legal_r = [_ler(p) for p in legal]
    bert_r = [_ler(p) for p in bertimbau]
    legal_f1 = [r["modelo"]["geral"]["f1"] for r in legal_r]
    bert_f1 = [r["modelo"]["geral"]["f1"] for r in bert_r]
    legal_media, bert_media = mean(legal_f1), mean(bert_f1)
    if legal_media <= 0 and bert_media <= 0:
        raise ValueError("os dois modelos colapsaram: F1 próprio igual a zero em todos os folds")
    vitorias = sum(a > b for a, b in zip(legal_f1, bert_f1))
    if bert_media <= 0:
        escolhido = "legal-bert"
    elif legal_media <= 0:
        escolhido = "bertimbau"
    else:
        consistente = vitorias >= 4 and legal_media > bert_media
        escolhido = "legal-bert" if consistente else "bertimbau"
    return {
        "modelo_escolhido": escolhido,
        "criterio": "F1 do modelo sem regras: Legal-BERT precisa vencer em >=4/5 folds e na média; caso contrário BERTimbau",
        "legal_bert": {"f1_modelo_folds": legal_f1, "media": legal_media, "desvio": pstdev(legal_f1), "vitorias": vitorias},
        "bertimbau": {"f1_modelo_folds": bert_f1, "media": bert_media, "desvio": pstdev(bert_f1)},
        "executar_teste_final_somente_com": escolhido,
    }


def _ler(pasta: Path) -> dict:
    return json.loads((pasta / "resultados_validacao.json").read_text(encoding="utf-8"))


def _manifesto(pasta: Path) -> dict:
    return json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legal", type=Path, nargs=5, required=True)
    parser.add_argument("--bertimbau", type=Path, nargs=5, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    resultado = selecionar(args.legal, args.bertimbau)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
