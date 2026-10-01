"""
Lookup isolado: mede só o índice e o resolvedor, sem extração nem normalização.

Para cada acórdão do índice, monta direto uma `CitacaoNormalizada` com a chave
que o próprio índice guarda (tribunal, classe, número, UF, origem) e pede ao
`Resolvedor` o registro correspondente. Não passa pelo extrator de spans nem
pela normalização de texto, então uma falha aqui é do índice ou da regra de
resolução, e não de etapas anteriores.

Complementa `teste_base_inteira.py`, que faz o caminho completo (frase →
extração → normalização → lookup). Comparando os dois, dá para ver em que etapa
um registro se perde: se o lookup isolado acerta e o caminho completo não, a
falha está antes do índice.

Desfechos por registro:

- `ok`: o resolvedor devolveu `real` com o id esperado;
- `ambiguo_com_esperado`: devolveu `incompleta` e o id esperado está entre os
  candidatos (resposta conservadora correta quando há duplicata no acervo);
- `ambiguo_sem_esperado`: devolveu `incompleta`, mas sem o id esperado;
- `real_errado`: devolveu `real` para outro id;
- `inventada`: não achou o registro que está no próprio índice.

São aceitáveis `ok` e `ambiguo_com_esperado`. `real_errado` é o erro grave: uma
resposta confiante e falsa.

Uso (da raiz do repositório):

    python src/validacao/lookup_isolado.py
    python src/validacao/lookup_isolado.py --exemplos 10
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eda_utils as u
import indice_utils as ind
from modelos import CitacaoNormalizada
from steps.buscar_na_base import Resolvedor
from validacao.teste_base_inteira import classe_para_citacao

DADOS = RAIZ / "data"
ACEITAVEIS = ("ok", "ambiguo_com_esperado")


def citacao_do_registro(tribunal: str, classe: str, numero: str, origem: str | None, uf: str | None) -> CitacaoNormalizada:
    """`CitacaoNormalizada` equivalente a citar o registro pela chave do índice."""
    classe_citada = classe_para_citacao(classe, tribunal)
    trecho = f"{classe_citada} nº {numero}" + (f"/{uf}" if uf else "")
    return CitacaoNormalizada(
        familia="acordao",
        inicio=0,
        fim=len(trecho),
        trecho=trecho,
        normalizado=trecho,
        tribunal=tribunal,
        classe=ind._classe_citada(classe_citada),
        numero=numero,
        uf=uf,
        origem=origem,
        identificador_suficiente=True,
    )


def avaliar_registro(resolvedor: Resolvedor, chave: tuple, registro: dict) -> tuple[str, str]:
    """(desfecho, caminho) do lookup isolado de um registro do índice."""
    tribunal, classe, numero, origem = chave
    citacao = citacao_do_registro(tribunal, registro["classe"], numero, origem, registro["uf"])
    resultado = resolvedor.resolver_citacao(citacao)
    esperado = registro["id"]
    if resultado.classe == "real":
        return ("ok" if resultado.id_canonico == esperado else "real_errado"), resultado.caminho
    if resultado.classe == "incompleta":
        return ("ambiguo_com_esperado" if esperado in resultado.candidatos else "ambiguo_sem_esperado"), resultado.caminho
    return "inventada", resultado.caminho


def executar(con: sqlite3.Connection, exemplos_max: int = 8) -> dict:
    resolvedor = Resolvedor(con)
    _, sem_parser = ind.construir_indice_acordaos(con)

    resultado = collections.Counter()
    por_tribunal = collections.Counter()
    falhas = collections.Counter()
    exemplos = []
    for chave, registros in resolvedor.acordaos.items():
        for registro in registros:
            desfecho, caminho = avaliar_registro(resolvedor, chave, registro)
            resultado[desfecho] += 1
            por_tribunal[(chave[0], desfecho)] += 1
            if desfecho not in ACEITAVEIS:
                falhas[(chave[0], desfecho, chave[1][:45], caminho)] += 1
                if len(exemplos) < exemplos_max:
                    exemplos.append((desfecho, chave, registro["id"]))

    return {
        "resultado": resultado,
        "por_tribunal": por_tribunal,
        "falhas": falhas,
        "exemplos": exemplos,
        "sem_parser": len(sem_parser),
        "numeros_distintos": len({c[2] for c in resolvedor.acordaos}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exemplos", type=int, default=8, help="registros de exemplo das falhas")
    args = parser.parse_args()

    con = sqlite3.connect(u._uri_sqlite((DADOS / "desafio1_bracis.db").resolve()) + "?immutable=1", uri=True)
    medicao = executar(con, args.exemplos)
    resultado = medicao["resultado"]
    total = sum(resultado.values())
    aceitaveis = sum(resultado[d] for d in ACEITAVEIS)

    print(f"Registros no índice: {total} (fora do índice: {medicao['sem_parser']} sem parser de cabeçalho)")
    print(f"Aceitáveis: {aceitaveis}/{total} ({aceitaveis / total:.1%})\n")
    print("Desfecho geral:")
    for desfecho in ("ok", "ambiguo_com_esperado", "ambiguo_sem_esperado", "real_errado", "inventada"):
        print(f"  {desfecho:22s} {resultado[desfecho]}")

    print("\nPor tribunal:")
    tribunais = sorted({t for t, _ in medicao["por_tribunal"]})
    for tribunal in tribunais:
        linha = ", ".join(f"{d} {n}" for (t, d), n in sorted(medicao["por_tribunal"].items()) if t == tribunal)
        print(f"  {tribunal}: {linha}")

    if medicao["falhas"]:
        print("\nFalhas (tribunal, desfecho, classe, caminho):")
        for chave, n in medicao["falhas"].most_common():
            print(f"  {n:4d}  {chave}")
        print("\nExemplos:")
        for desfecho, chave, id_esperado in medicao["exemplos"]:
            print(f"  [{desfecho}] id {id_esperado} {chave}")


if __name__ == "__main__":
    main()
