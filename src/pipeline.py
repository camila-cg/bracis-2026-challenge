"""
Pipeline de ponta a ponta — Desafio 1, BRACIS 2026 / Jusbrasil.

Etapas, na ordem em que cada citação passa por elas (uma por módulo em src/steps/):

    Por documento (processar_documento):
      1. extrair               txt -> candidatos a citação (detectar_citacoes)
      2. filtrar_distratores   descarta R$, fls., OAB, autos do próprio processo
      3. rotear                decide o tipo: lei, súmula, tema, descritiva, acórdão
      4. normalizar            extrai atributos e corrige OCR em CitacaoNormalizada
      5. resolver              consulta por atributos -> ResultadoResolucao
      6. atribuir_confianca    confiança pelo caminho de decisão da etapa 5

    Para o conjunto (main):
      7. gravar_saida          um JSON por documento -> submission.csv
      +  avaliar (opcional)    pontua com data/kaggle_metric.py (--avaliar)

Uso (da raiz do repositório):

    python src/pipeline.py                       # data/txt -> saida/json + saida/submission.csv
    python src/pipeline.py --avaliar             # idem, e pontua com data/kaggle_metric.py
    python src/pipeline.py --txt <pasta> --saida <pasta>

Nota: Nada aqui olha o gabarito: --avaliar só lê o goldenset depois que as predições estão prontas.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eda_utils as u
import indice_utils as ind  # noqa: F401 — usado por src/validacao via p.ind
from steps.pipeline_utils import DADOS, RAIZ
from steps.extrair_candidatos import extrair                  # 1
from steps.filtrar_distratores import filtrar_distratores     # 2
from steps.rotear_citacao import rotear                       # 3
from steps.normalizar_citacao import normalizar_citacao        # 4
from steps.buscar_na_base import Resolvedor                    # 5
from steps.atribuir_confianca import atribuir_confianca       # 6
from steps.salvar_submissao import gravar_saida               # 7
from steps.avaliar_metrica import avaliar, imprimir_avaliacao


def ler_texto(arquivo: Path) -> str:
    bruto = arquivo.read_bytes()
    if b"\r\n" in bruto:
        raise ValueError(f"{arquivo.name} contém CRLF; os offsets seriam deslocados.")
    return bruto.decode("utf-8")


def processar_documento(documento_id: str, texto: str, resolvedor: Resolvedor) -> dict:
    candidatos = extrair(texto)                                        # 1
    candidatos = filtrar_distratores(candidatos, texto)                # 2
    citacoes = []
    for candidato in candidatos:
        rota = rotear(candidato)                                       # 3
        citacao = normalizar_citacao(candidato, rota)                   # 4
        resultado = resolvedor.resolver_citacao(citacao)               # 5
        confianca = atribuir_confianca(resultado.caminho, citacao.fonte)  # 6
        citacoes.append({
            "inicio": candidato["inicio"],
            "fim": candidato["fim"],
            "trecho": candidato["trecho"],
            "tipo": candidato["tipo"],
            "classificacao": resultado.classe,
            "resolucao": {"id_canonico": str(resultado.id_canonico)} if resultado.id_canonico is not None else None,
            "confianca": confianca,
            "caminho": resultado.caminho,
            "cobertura": resultado.cobertura,
            "motivo": resultado.motivo,
            "fonte": citacao.fonte,
        })
    return {"documento_id": documento_id, "citacoes": citacoes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--txt", type=Path, default=DADOS / "txt")
    parser.add_argument("--db", type=Path, default=DADOS / "desafio1_bracis.db")
    parser.add_argument("--saida", type=Path, default=RAIZ / "saida")
    parser.add_argument("--avaliar", action="store_true", help="pontua contra data/goldenset_offsets.csv")
    parser.add_argument("--goldenset", type=Path, default=DADOS / "goldenset_offsets.csv")
    args = parser.parse_args()

    arquivos = sorted(args.txt.glob("*.txt"))
    if not arquivos:
        sys.exit(f"nenhum .txt em {args.txt}")
    con = sqlite3.connect(u._uri_sqlite(args.db.resolve()) + "?immutable=1", uri=True)
    resolvedor = Resolvedor(con)

    documentos = [processar_documento(a.stem, ler_texto(a), resolvedor) for a in arquivos]  # 1–6
    submission_csv = gravar_saida(documentos, args.saida)                                   # 7

    if args.avaliar:
        imprimir_avaliacao(avaliar(submission_csv, args.goldenset))


if __name__ == "__main__":
    main()
