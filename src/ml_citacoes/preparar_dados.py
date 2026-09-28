"""CLI da preparação do dataset original com sintéticos opcionais no treino."""

from __future__ import annotations

import argparse
from pathlib import Path

from .splits import preparar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=Path, required=True)
    parser.add_argument("--goldenset", type=Path, required=True)
    parser.add_argument(
        "--sintetico",
        type=Path,
        action="append",
        default=[],
        help="pacote sintético com manifest.csv, goldenset_offsets.csv e txt/; pode ser repetido",
    )
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260926)
    args = parser.parse_args()
    preparar(
        args.txt,
        args.goldenset,
        args.saida,
        n_folds=args.folds,
        seed=args.seed,
        sinteticos=args.sintetico,
    )


if __name__ == "__main__":
    main()
