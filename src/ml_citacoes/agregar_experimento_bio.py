"""Agrega a ablação pareada dos cinco folds sem ler textos ou erros locais."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def _estatisticas(valores: list[float]) -> dict:
    return {
        "valores": valores,
        "media": mean(valores),
        "desvio_populacional": pstdev(valores),
    }


def _prf(resumo: dict, modo: str, grupo: str, metrica: str = "iou_05") -> dict:
    return resumo["variantes"][modo][grupo][metrica]["geral"]


def agregar(caminhos: list[Path]) -> dict:
    if len(caminhos) != 5:
        raise ValueError("a agregação exige exatamente cinco resumos")
    resumos = [json.loads(caminho.read_text(encoding="utf-8")) for caminho in caminhos]
    por_fold = {int(resumo["fold"]): resumo for resumo in resumos}
    if set(por_fold) != set(range(5)) or len(por_fold) != 5:
        raise ValueError("os resumos precisam cobrir uma vez cada fold de 0 a 4")
    ordenados = [por_fold[i] for i in range(5)]

    invariantes = (
        "dataset_sha256", "codigo_sha256", "regras_config_sha256",
        "limiares_fixos", "calibracao", "ocr_definicao",
    )
    for chave in invariantes:
        if any(item[chave] != ordenados[0][chave] for item in ordenados[1:]):
            raise ValueError(f"folds executados com {chave} divergente")
    for item in ordenados:
        reproducao = item.get("reproducao_argmax")
        if not reproducao or reproducao["documentos_com_spans_diferentes"] != 0:
            raise ValueError(f"fold {item['fold']} não reproduziu o argmax original")
        if item["variantes"]["argmax"]["colapso"] or item["variantes"]["viterbi"]["colapso"]:
            raise ValueError(f"fold {item['fold']} colapsado")

    saida = {
        "protocolo": "exp01_bio_5folds_pareado",
        "avaliacao_independente": False,
        "motivo": "folds usados para escolher checkpoints; comparação do decoder é pós-hoc",
        "folds": list(range(5)),
        "dataset_sha256": ordenados[0]["dataset_sha256"],
        "limiares_fixos": ordenados[0]["limiares_fixos"],
        "n_documentos_total": sum(item["n_documentos"] for item in ordenados),
        "n_entidades_total": sum(item["n_entidades"] for item in ordenados),
        "tokens_alterados": {
            "por_fold": [item["tokens_alterados"] for item in ordenados],
            "total": sum(item["tokens_alterados"] for item in ordenados),
        },
        "regras_recall_perfeito_folds": sum(item["regras_perfeitas_em_recall"] for item in ordenados),
        "variantes": {},
    }
    for modo in ("argmax", "viterbi"):
        saida["variantes"][modo] = {}
        for grupo in ("bruto", "limiar", "fusao"):
            bloco = {}
            for nome_metrica in ("iou_05", "exato"):
                gerais = [_prf(item, modo, grupo, nome_metrica) for item in ordenados]
                bloco[nome_metrica] = {
                    chave: _estatisticas([float(m[chave]) for m in gerais])
                    for chave in ("precisao", "recall", "f1")
                }
                bloco[nome_metrica]["totais"] = {
                    chave: sum(int(m[chave]) for m in gerais) for chave in ("tp", "fp", "fn")
                }
            saida["variantes"][modo][grupo] = bloco
        diagnosticos = [item["variantes"][modo]["diagnostico_bio"] for item in ordenados]
        saida["variantes"][modo]["diagnostico_bio"] = {
            "transicoes_invalidas_total": sum(d["transicoes_invalidas"] for d in diagnosticos),
            "documentos_invalidos_total": sum(d["documentos_invalidos"] for d in diagnosticos),
            "taxa_documentos_invalidos": _estatisticas(
                [d["taxa_documentos_invalidos"] for d in diagnosticos]
            ),
        }
        saida["variantes"][modo]["bruto_por_tipo"] = {
            tipo: {
                chave: _estatisticas([
                    item["variantes"][modo]["bruto"]["iou_05"]["por_tipo"][tipo][chave]
                    for item in ordenados
                ])
                for chave in ("precisao", "recall", "f1")
            }
            for tipo in ("LEI", "JUR")
        }

    saida["delta_viterbi_menos_argmax"] = {
        grupo: {
            chave: _estatisticas([
                item["delta_viterbi_menos_argmax"][grupo][chave] for item in ordenados
            ])
            for chave in ("tp", "fp", "fn", "precisao", "recall", "f1")
        }
        for grupo in ("bruto", "limiar", "fusao")
    }
    return saida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resumos", type=Path, nargs=5)
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args()
    resultado = agregar(args.resumos)
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.saida:
        if args.saida.exists():
            raise ValueError("saída agregada já existe")
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(texto + "\n", encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
