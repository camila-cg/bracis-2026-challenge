"""
Teste de cobertura sobre a base canônica inteira.

Para cada acórdão que o índice de cabeçalhos consegue ler, escreve uma frase
com uma citação simples a ele ("... no {classe} nº {número}/{UF}, ...") e
passa pelo pipeline. Mede quantos registros o sistema consegue extrair e
resolver para o `id` certo, e agrupa as falhas por tribunal e classe.

Serve para estimar como o pipeline se sairia com citações a registros que
o dev set não cobre. É uma aproximação: a classe entra por extenso, como
está no cabeçalho, e não na forma abreviada que um documento real usaria.

Uso (da raiz do repositório):

    python src/validacao/teste_base_inteira.py
    python src/validacao/teste_base_inteira.py --top 40 --exemplos 10
"""

from __future__ import annotations

import argparse
import collections
import re
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eda_utils as u
import pipeline as p

DADOS = RAIZ / "data"

# Órgão julgador no início da classe do cabeçalho ("PRIMEIRA TURMA AG.REG. ...").
ORGAO = re.compile(r"^(PRIMEIRA|SEGUNDA|TERCEIRA|QUARTA|QUINTA|SEXTA)\s+TURMA\s+|^PLEN[ÁA]RIO\s+", re.I)


def frase_de_citacao(classe: str, numero: str, uf: str | None) -> str:
    classe = ORGAO.sub("", classe).title()
    numero = numero if "-" in numero else u.agrupar_milhares(numero)
    return (f"Nesse sentido, confira-se o que decidiu esta Corte no {classe} nº "
            f"{numero}/{uf or 'DF'}, cujo entendimento se aplica ao caso.\n")


def avaliar_registro(texto: str, id_esperado: int, resolvedor: p.Resolvedor) -> tuple[str, str]:
    citacoes = p.processar_documento("sintetico", texto, resolvedor)["citacoes"]
    if not citacoes:
        return "nao_extraiu", "-"
    c = max(citacoes, key=lambda c: c["fim"] - c["inicio"])
    id_previsto = (c["resolucao"] or {}).get("id_canonico")
    if id_previsto == str(id_esperado):
        return "ok", c["caminho"]
    if c["classificacao"] == "incompleta":
        return "incompleta", c["caminho"]
    return f"{c['classificacao']}_errado", c["caminho"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=25, help="grupos de falha a listar")
    parser.add_argument("--exemplos", type=int, default=8, help="frases de exemplo das falhas")
    args = parser.parse_args()

    con = sqlite3.connect(u._uri_sqlite((DADOS / "desafio1_bracis.db").resolve()) + "?immutable=1", uri=True)
    resolvedor = p.Resolvedor(con)
    _, sem_parser = p.ind.construir_indice_acordaos(con)

    resultado = collections.Counter()
    por_tribunal = collections.Counter()
    falhas = collections.Counter()
    exemplos = []

    for (tribunal, _classe, numero, _origem), registros in resolvedor.acordaos.items():
        for r in registros:
            texto = frase_de_citacao(r["classe"], numero, r["uf"])
            desfecho, caminho = avaliar_registro(texto, r["id"], resolvedor)
            resultado[desfecho] += 1
            por_tribunal[(tribunal, desfecho)] += 1
            if desfecho != "ok":
                falhas[(tribunal, desfecho, ORGAO.sub("", r["classe"])[:45], caminho)] += 1
                if len(exemplos) < args.exemplos:
                    exemplos.append((desfecho, texto.strip()[:120]))

    total = sum(resultado.values())
    print(f"Registros testados: {total} (fora do teste, sem parser de cabeçalho: {len(sem_parser)})")
    print(f"Resolvidos corretamente: {resultado['ok']} ({resultado['ok'] / total:.1%})\n")

    print("Desfecho geral:")
    for desfecho, n in resultado.most_common():
        print(f"  {desfecho:18s} {n}")

    print("\nPor tribunal:")
    for tribunal in sorted({t for t, _ in por_tribunal}):
        linha = ", ".join(f"{d} {n}" for (t, d), n in sorted(por_tribunal.items()) if t == tribunal)
        print(f"  {tribunal}: {linha}")

    print(f"\nPrincipais falhas (tribunal, desfecho, classe, caminho):")
    for chave, n in falhas.most_common(args.top):
        print(f"  {n:4d}  {chave}")

    print("\nExemplos:")
    for desfecho, frase in exemplos:
        print(f"  [{desfecho}] {frase}")


if __name__ == "__main__":
    main()
