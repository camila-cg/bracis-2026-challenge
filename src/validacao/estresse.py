"""
Teste de estresse: perturba as citações do gabarito e mede o pipeline com a
métrica oficial.

Cenários (cada um aplicado sobre os 26 documentos):
  original - texto como distribuído (sanidade: deve bater com o --avaliar)
  ruido    - ruído de superfície do nível 2 nas citações reais de acórdão
             (abreviação, formato do número, separador de UF, OCR, quebra
             de linha); a classe esperada continua `real`
  digito   - troca um dígito do identificador das reais de acórdão; a classe
             esperada passa a ser `inventada`

Uso (da raiz do repositório):

    python src/validacao/estresse.py
    python src/validacao/estresse.py --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import re
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import eda_utils as u
import pipeline as p

DADOS = RAIZ / "data"

ABREVIACOES = [
    ("Recurso Especial", ["REsp", "R.Esp.", "Rec. Esp."]),
    ("REsp", ["Recurso Especial", "Rec. Esp.", "R. Esp."]),
    ("AREsp", ["Agravo em Recurso Especial", "Ag. em REsp"]),
    ("Habeas Corpus", ["HC"]),
    ("HC", ["Habeas Corpus", "H.C."]),
    ("Recurso Extraordinário", ["RE"]),
    ("Reclamação", ["Rcl", "Recl."]),
    ("Rcl", ["Reclamação", "Recl."]),
    ("nº", ["n.", "No", "n°", "número"]),
]
# Só confusões letra<->dígito: o desafio garante que dígito nunca vira dígito.
OCR = {"0": "O", "1": "l", "5": "S"}


def carregar_modulo(caminho: Path, nome: str):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def aplicar_ruido(trecho: str, rng: random.Random) -> str:
    t = trecho
    for original, variantes in ABREVIACOES:
        if original in t and rng.random() < 0.6:
            t = t.replace(original, rng.choice(variantes), 1)

    def formato(m: re.Match) -> str:
        s = m.group(0)
        op = rng.choice(["nada", "sem_ponto", "espaco", "quebra"])
        if op == "sem_ponto":
            return s.replace(".", "")
        if op == "espaco":
            return s.replace(".", ". ", 1)
        if op == "quebra":
            return s.replace(".", ".\n", 1)
        return s

    t = re.sub(r"\d{1,3}(?:\.\d{3})+", formato, t)
    t = re.sub(r"/([A-Z]{2})\b",
               lambda m: rng.choice([f"/{m.group(1)}", f" - {m.group(1)}", f" ({m.group(1)})"]), t)
    posicoes = [i for i, c in enumerate(t) if c in OCR]
    if posicoes and rng.random() < 0.7:
        i = rng.choice(posicoes)
        t = t[:i] + OCR[t[i]] + t[i + 1:]
    return t


def trocar_digito(trecho: str, rng: random.Random) -> str | None:
    """Troca um dígito do bloco numérico mais longo (o identificador)."""
    blocos = list(re.finditer(r"\d[\d.\-/]*\d", trecho))
    if not blocos:
        return None
    bloco = max(blocos, key=lambda m: len(m.group(0)))
    posicoes = [bloco.start() + i for i, c in enumerate(bloco.group(0)) if c.isdigit()]
    i = rng.choice(posicoes)
    novo = rng.choice([d for d in "0123456789" if d != trecho[i]])
    return trecho[:i] + novo + trecho[i + 1:]


def carregar_gabarito() -> dict[str, list[dict]]:
    gabarito: dict[str, list[dict]] = {}
    with (DADOS / "goldenset_offsets.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            gabarito.setdefault(r["documento_id"], []).append(r)
    return gabarito


def montar_cenario(cenario: str, textos: dict, gabarito: dict, rng: random.Random):
    """Reescreve os textos e recalcula os offsets do gabarito."""
    documentos, solucao = {}, []
    for doc_id, texto in textos.items():
        citacoes = sorted(gabarito.get(doc_id, []), key=lambda r: int(r["inicio"]))
        partes, pos, novas = [], 0, []
        for r in citacoes:
            inicio, fim = int(r["inicio"]), int(r["fim"])
            trecho = texto[inicio:fim]
            classe, id_canonico = r["classificacao"], r["id_canonico"] or "-"
            e_acordao_real = (classe == "real" and r["tipo"] == "jurisprudencia"
                              and not re.search(r"[Ss]úmula", trecho))
            if e_acordao_real and cenario == "ruido":
                trecho = aplicar_ruido(trecho, rng)
            elif e_acordao_real and cenario == "digito":
                alterado = trocar_digito(trecho, rng)
                if alterado:
                    trecho, classe, id_canonico = alterado, "inventada", "-"
            partes.append(texto[pos:inicio])
            novo_inicio = sum(map(len, partes))
            partes.append(trecho)
            novas.append(f"{novo_inicio},{novo_inicio + len(trecho)},{classe},{id_canonico}")
            pos = fim
        partes.append(texto[pos:])
        documentos[doc_id] = "".join(partes)
        solucao.append({"documento_id": doc_id, "nivel": u.nivel(doc_id),
                        "citacoes": "|".join(novas) or "-"})
    return documentos, pd.DataFrame(solucao)


def predizer(documentos: dict, resolvedor: p.Resolvedor) -> pd.DataFrame:
    linhas = []
    for doc_id, texto in documentos.items():
        citacoes = p.processar_documento(doc_id, texto, resolvedor)["citacoes"]
        linhas.append({"documento_id": doc_id, "citacoes": "|".join(
            f"{c['inicio']},{c['fim']},{c['classificacao']},"
            f"{(c['resolucao'] or {}).get('id_canonico', '-')},{c['confianca']}"
            for c in citacoes) or "-"})
    return pd.DataFrame(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3, help="repetições por cenário perturbado")
    args = parser.parse_args()

    metrica = carregar_modulo(DADOS / "kaggle_metric.py", "kaggle_metric")
    gabarito = carregar_gabarito()
    textos = {f.stem: p.ler_texto(f) for f in sorted((DADOS / "txt").glob("*.txt"))}
    con = sqlite3.connect(u._uri_sqlite((DADOS / "desafio1_bracis.db").resolve()) + "?immutable=1", uri=True)
    resolvedor = p.Resolvedor(con)

    for cenario in ["original", "ruido", "digito"]:
        for seed in range(1 if cenario == "original" else args.seeds):
            documentos, solucao = montar_cenario(cenario, textos, gabarito, random.Random(seed))
            r = metrica.avaliar(solucao, predizer(documentos, resolvedor), row_id="documento_id")
            niveis = " | ".join(
                f"N{n}: F1 {v['macro_f1']:.3f} τ {v['tau']:.2f} "
                f"({', '.join(f'{c[:4]} {x:.2f}' for c, x in v['f1_por_classe'].items())})"
                for n, v in r["niveis"].items())
            print(f"{cenario:9s} seed{seed}: final {r['score_final']:.4f}  {niveis}")


if __name__ == "__main__":
    main()
