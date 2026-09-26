"""
Prepara uma submissão para o Kaggle: valida o submission.csv e empacota o bundle.

Gera, em saida/submissao_<commit>/:
    submission.csv   cópia validada do CSV (pode subir direto no Kaggle)
    submission.zip   o mesmo CSV zipado (o Kaggle aceita os dois)
    bundle_<commit>.zip  o código no commit exato (git archive) + comando.txt

Validações do CSV: cabeçalho documento_id,citacoes; uma linha por .txt de
entrada, sem faltar nem sobrar; nenhuma célula vazia; cada citação no formato
inicio,fim,classe,id_canonico,confianca com classe válida.

O bundle só é gerado com a árvore de trabalho limpa (tudo commitado), senão o
commit do nome não reproduziria as saídas. Use --permitir-sujo para testes.

Uso (da raiz do repositório, depois de rodar src/pipeline.py):

    python src/empacotar_submissao.py
    python src/empacotar_submissao.py --csv saida/submission.csv --txt data/txt
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parents[1]
CLASSES = {"real", "inventada", "incompleta"}
CITACAO = re.compile(r"^(\d+),(\d+),([a-z]+),([^,|]+),(-|\d+(?:\.\d+)?)$")
COMANDO = "uv sync --no-default-groups && uv run --no-default-groups python src/pipeline.py"


def validar(csv_path: Path, pasta_txt: Path) -> list[str]:
    erros = []
    with csv_path.open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.reader(arquivo))
    if not linhas or linhas[0] != ["documento_id", "citacoes"]:
        return [f"cabeçalho deve ser documento_id,citacoes (veio {linhas[:1]})"]

    vistos = []
    for n, linha in enumerate(linhas[1:], start=2):
        if len(linha) != 2:
            erros.append(f"linha {n}: esperadas 2 colunas, vieram {len(linha)}")
            continue
        documento_id, citacoes = linha
        vistos.append(documento_id)
        if not citacoes:
            erros.append(f"linha {n} ({documento_id}): célula vazia; use '-'")
        elif citacoes != "-":
            for citacao in citacoes.split("|"):
                m = CITACAO.match(citacao)
                if not m:
                    erros.append(f"linha {n} ({documento_id}): citação mal formada: {citacao!r}")
                elif m[3] not in CLASSES:
                    erros.append(f"linha {n} ({documento_id}): classe inválida {m[3]!r}")
                elif int(m[1]) >= int(m[2]):
                    erros.append(f"linha {n} ({documento_id}): inicio >= fim em {citacao!r}")

    esperados = {p.stem for p in pasta_txt.glob("*.txt")}
    if not esperados:
        erros.append(f"nenhum .txt em {pasta_txt} para conferir os documento_id")
    duplicados = {d for d in vistos if vistos.count(d) > 1}
    if duplicados:
        erros.append(f"documento_id duplicados: {sorted(duplicados)}")
    if faltando := esperados - set(vistos):
        erros.append(f"faltam documentos: {sorted(faltando)}")
    if sobrando := set(vistos) - esperados if esperados else set():
        erros.append(f"documentos que não estão em {pasta_txt}: {sorted(sobrando)}")
    return erros


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, check=True, capture_output=True, text=True).stdout.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=RAIZ / "saida" / "submission.csv")
    parser.add_argument("--txt", type=Path, default=RAIZ / "data" / "txt")
    parser.add_argument("--permitir-sujo", action="store_true", help="gera o bundle mesmo com mudanças não commitadas")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"{args.csv} não existe; rode antes: python src/pipeline.py")
    if erros := validar(args.csv, args.txt):
        sys.exit("submission.csv inválido:\n  " + "\n  ".join(erros))
    print(f"{args.csv}: OK")

    sujo = git("status", "--porcelain", "--untracked-files=no")
    if sujo and not args.permitir_sujo:
        sys.exit("há mudanças não commitadas em arquivos rastreados; commite antes "
                 "para o bundle corresponder ao commit (ou use --permitir-sujo):\n" + sujo)
    commit = git("rev-parse", "--short", "HEAD")

    destino = RAIZ / "saida" / f"submissao_{commit}"
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.csv, destino / "submission.csv")
    with zipfile.ZipFile(destino / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.csv, "submission.csv")

    bundle = destino / f"bundle_{commit}.zip"
    git("archive", "--format=zip", f"--prefix=bracis-2026-challenge-{commit}/", "-o", str(bundle), "HEAD")
    with zipfile.ZipFile(bundle, "a") as z:
        z.writestr(f"bracis-2026-challenge-{commit}/comando.txt",
                   f"commit: {git('rev-parse', 'HEAD')}\n"
                   f"dados: coloque txt/ e desafio1_bracis.db em data/ (ver README)\n"
                   f"comando: {COMANDO}\n"
                   f"saída: saida/submission.csv\n")

    print(f"pronto em {destino}:")
    print(f"  submission.csv / submission.zip -> subir no Kaggle")
    print(f"  {bundle.name} -> bundle de reprodutibilidade" + (" (ATENÇÃO: árvore suja)" if sujo else ""))


if __name__ == "__main__":
    main()
