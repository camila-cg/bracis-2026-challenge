"""Gera, opcionalmente, um ZIP contendo o CSV produzido pelo pipeline.

Uso (da raiz do repositório, depois de rodar ``src/pipeline.py``):

    python src/empacotar_submissao.py
    python src/empacotar_submissao.py --csv saida/submission.csv --saida saida/submission.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parents[1]
def gerar_zip(csv_path: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as arquivo:
        arquivo.write(csv_path, "submission.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=RAIZ / "saida" / "submission.csv")
    parser.add_argument("--saida", type=Path, default=RAIZ / "saida" / "submission.zip",
                        help="ZIP opcional a gerar (padrão: saida/submission.zip)")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"{args.csv} não existe; rode antes: python src/pipeline.py")
    gerar_zip(args.csv, args.saida)
    print(f"ZIP gerado: {args.saida}")


if __name__ == "__main__":
    main()
