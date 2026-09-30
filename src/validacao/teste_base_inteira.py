"""
Teste de cobertura sobre a base canônica inteira.

Para cada acórdão que o índice de cabeçalhos consegue ler, escreve uma frase
com uma citação simples a ele ("... no {classe} nº {número}/{UF}, ...") e
passa pelo pipeline. Mede quantos registros o sistema consegue extrair e
resolver para o `id` certo, e agrupa as falhas por tribunal e classe.

Serve para estimar como o pipeline se sairia com citações a registros que
o dev set não cobre. É uma aproximação: a classe entra por extenso, como
está no cabeçalho, e não na forma abreviada que um documento real usaria.

A frase procura se parecer com uma citação real:

- a classe perde o ruído do cabeçalho do TSE ("Tribunal Superior Eleitoral
  Acórdão …", "Publicado em Sessão …"), que nenhuma citação traria;
- o número CNJ entra pontuado ("0600393-67.2020.6.06.0075"), não como a chave
  interna do índice;
- a UF só aparece quando o cabeçalho a tem.

Registros cujo número saiu corrompido do cabeçalho (só dígitos, por OCR) não
entram na conta: não há citação realista a gerar para eles. Ficam contados à
parte (ver A3 no diagnóstico).

Registros que dividem número e classe com outro do mesmo tribunal são duplicatas
do acervo: nenhuma citação os distingue, então `incompleta` é a resposta certa.
Contam como `duplicata_legitima`, não como falha. Em TSE, TST e STM, todo número
com mais de um registro cai nesse caso.

O teste também mede falsos positivos: conta as citações extras na própria frase
e as citações detectadas sobre os distratores de `data/sinteticos_desafio`.

Uso (da raiz do repositório):

    python src/validacao/teste_base_inteira.py
    python src/validacao/teste_base_inteira.py --top 40 --exemplos 10
"""

from __future__ import annotations

import argparse
import collections
import json
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
SINTETICOS = DADOS / "sinteticos_desafio"

# Órgão julgador no início da classe do cabeçalho ("PRIMEIRA TURMA AG.REG. ...").
ORGAO = re.compile(r"^(PRIMEIRA|SEGUNDA|TERCEIRA|QUARTA|QUINTA|SEXTA)\s+TURMA\s+|^PLEN[ÁA]RIO\s+", re.I)

# Ruído que o parser do TSE deixa no começo da classe: tudo até "ACÓRDÃO",
# ou o nome do tribunal sem o "ACÓRDÃO", ou um "«" de OCR.
RUIDO_DE_CABECALHO = re.compile(
    r"^(?:.*?\bAC[ÓO]RD[ÃA]O\s+"
    r"|(?:PUBLICADO\s+EM\s+SESS[ÃA]O\s+)?(?:TRIBUNAL\.?\s+)?SUPERIOR\s+ELEITORAL\s+"
    r"|«\s*)",
    re.I,
)
CHAVE_CNJ = re.compile(r"^\d{7}-\d{13}$")


def classe_para_citacao(classe: str, tribunal: str) -> str:
    limpa = RUIDO_DE_CABECALHO.sub("", ORGAO.sub("", classe)).strip()
    # Os códigos do TST (RR, AIRR, Ag-AIRR) são siglas: "Rr" não é como se cita.
    return limpa if tribunal == "TST" else limpa.title()


def numero_corrompido(numero: str) -> bool:
    """Só dígitos e longo demais para ser número comum: CNJ que perdeu a estrutura."""
    return len(numero) >= 14 and not CHAVE_CNJ.match(numero) and "-" not in numero


def numero_para_citacao(numero: str) -> str:
    if CHAVE_CNJ.match(numero):
        return u.formatar_cnj_para_fts(numero)
    return numero if "-" in numero else u.agrupar_milhares(numero)


def frase_de_citacao(classe: str, numero: str, uf: str | None, tribunal: str = "") -> str:
    identificador = numero_para_citacao(numero) + (f"/{uf}" if uf else "")
    return (f"Nesse sentido, confira-se o que decidiu esta Corte no "
            f"{classe_para_citacao(classe, tribunal)} nº {identificador}, cujo entendimento se aplica ao caso.\n")


def avaliar_registro(texto: str, id_esperado: int, resolvedor: p.Resolvedor) -> tuple[str, str, int]:
    """(desfecho, caminho, citações extras além da escolhida)."""
    citacoes = p.processar_documento("sintetico", texto, resolvedor)["citacoes"]
    if not citacoes:
        return "nao_extraiu", "-", 0
    extras = len(citacoes) - 1
    c = max(citacoes, key=lambda c: c["fim"] - c["inicio"])
    id_previsto = (c["resolucao"] or {}).get("id_canonico")
    if id_previsto == str(id_esperado):
        return "ok", c["caminho"], extras
    if c["classificacao"] == "incompleta":
        return "incompleta", c["caminho"], extras
    return f"{c['classificacao']}_errado", c["caminho"], extras


def ids_duplicados(resolvedor: p.Resolvedor) -> set[int]:
    """Ids que dividem número e classe com outro registro do mesmo tribunal.

    Nenhuma citação os distingue (número e classe são iguais), então `incompleta`
    é a resposta correta para eles, não uma falha do resolvedor.
    """
    grupos: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for (tribunal, _classe, numero, _origem), registros in resolvedor.acordaos.items():
        grupos[(tribunal, numero)].extend(registros)
    duplicados: set[int] = set()
    for (tribunal, _numero), registros in grupos.items():
        if len(registros) < 2:
            continue
        classes = [classe_para_citacao(r["classe"], tribunal).upper() for r in registros]
        contagem = collections.Counter(classes)
        duplicados |= {r["id"] for r, classe in zip(registros, classes) if contagem[classe] > 1}
    return duplicados


def falsos_positivos_em_distratores(resolvedor: p.Resolvedor) -> tuple[int, int]:
    """(citações detectadas sobre um distrator, total de distratores)."""
    distratores: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    with (SINTETICOS / "distratores.jsonl").open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            if linha.strip():
                d = json.loads(linha)
                distratores[d["documento_id"]].append((d["inicio"], d["fim"]))
    detectados = 0
    with (SINTETICOS / "documentos.jsonl").open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            if not linha.strip():
                continue
            documento = json.loads(linha)
            spans = distratores.get(documento["documento_id"], [])
            if not spans:
                continue
            citacoes = p.processar_documento(documento["documento_id"], documento["texto"], resolvedor)["citacoes"]
            for c in citacoes:
                if any(c["inicio"] < fim and inicio < c["fim"] for inicio, fim in spans):
                    detectados += 1
    return detectados, sum(len(spans) for spans in distratores.values())


def executar(con: sqlite3.Connection, exemplos_max: int = 8) -> dict:
    resolvedor = p.Resolvedor(con)
    _, sem_parser = p.ind.construir_indice_acordaos(con)

    resultado = collections.Counter()
    por_tribunal = collections.Counter()
    falhas = collections.Counter()
    exemplos = []
    corrompidas = collections.Counter()
    extras_total = 0
    duplicados = ids_duplicados(resolvedor)

    for (tribunal, _classe, numero, _origem), registros in resolvedor.acordaos.items():
        for r in registros:
            if numero_corrompido(numero):
                corrompidas[tribunal] += 1
                continue
            texto = frase_de_citacao(r["classe"], numero, r["uf"], tribunal)
            desfecho, caminho, extras = avaliar_registro(texto, r["id"], resolvedor)
            extras_total += extras
            if desfecho == "incompleta" and r["id"] in duplicados:
                desfecho = "duplicata_legitima"
            resultado[desfecho] += 1
            por_tribunal[(tribunal, desfecho)] += 1
            if desfecho != "ok":
                falhas[(tribunal, desfecho, ORGAO.sub("", r["classe"])[:45], caminho)] += 1
                if len(exemplos) < exemplos_max:
                    exemplos.append((desfecho, texto.strip()[:120]))

    fp_distratores, total_distratores = falsos_positivos_em_distratores(resolvedor)
    return {
        "resultado": resultado,
        "por_tribunal": por_tribunal,
        "falhas": falhas,
        "exemplos": exemplos,
        "corrompidas": corrompidas,
        "extras": extras_total,
        "sem_parser": len(sem_parser),
        "fp_distratores": fp_distratores,
        "total_distratores": total_distratores,
    }


def ok_por_tribunal(medicao: dict) -> dict[str, tuple[int, int]]:
    """{tribunal: (corretos, avaliáveis)}; duplicatas legítimas não são avaliáveis."""
    contagem: dict[str, list[int]] = {}
    for (tribunal, desfecho), n in medicao["por_tribunal"].items():
        acumulado = contagem.setdefault(tribunal, [0, 0])
        if desfecho != "duplicata_legitima":
            acumulado[1] += n
        if desfecho == "ok":
            acumulado[0] += n
    return {tribunal: (ok, total) for tribunal, (ok, total) in contagem.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=25, help="grupos de falha a listar")
    parser.add_argument("--exemplos", type=int, default=8, help="frases de exemplo das falhas")
    args = parser.parse_args()

    con = sqlite3.connect(u._uri_sqlite((DADOS / "desafio1_bracis.db").resolve()) + "?immutable=1", uri=True)
    medicao = executar(con, args.exemplos)
    resultado = medicao["resultado"]
    total = sum(resultado.values())

    print(f"Registros testados: {total} (fora do teste: {medicao['sem_parser']} sem parser de cabeçalho "
          f"e {sum(medicao['corrompidas'].values())} com número corrompido no cabeçalho)")
    duplicatas = resultado["duplicata_legitima"]
    print(f"Resolvidos corretamente: {resultado['ok']} ({resultado['ok'] / total:.1%} do total; "
          f"{resultado['ok'] / (total - duplicatas):.1%} sem as duplicatas legítimas)")
    print(f"Duplicatas legítimas (incompleta é a resposta certa): {duplicatas}\n")

    print("Desfecho geral:")
    for desfecho, n in resultado.most_common():
        print(f"  {desfecho:18s} {n}")

    print("\nPor tribunal:")
    for tribunal, (ok, testados) in sorted(ok_por_tribunal(medicao).items()):
        linha = ", ".join(f"{d} {n}" for (t, d), n in sorted(medicao["por_tribunal"].items()) if t == tribunal)
        print(f"  {tribunal}: {ok}/{testados} ({ok / testados:.1%}) — {linha}")

    if medicao["corrompidas"]:
        print("\nNúmero corrompido no cabeçalho (fora da conta, ver A3): "
              + ", ".join(f"{t} {n}" for t, n in sorted(medicao["corrompidas"].items())))

    print("\nFalsos positivos:")
    print(f"  citações extras dentro da própria frase: {medicao['extras']}")
    print(f"  citações detectadas sobre distratores:   {medicao['fp_distratores']} de {medicao['total_distratores']}")

    print(f"\nPrincipais falhas (tribunal, desfecho, classe, caminho):")
    for chave, n in medicao["falhas"].most_common(args.top):
        print(f"  {n:4d}  {chave}")

    print("\nExemplos:")
    for desfecho, frase in medicao["exemplos"]:
        print(f"  [{desfecho}] {frase}")


if __name__ == "__main__":
    main()
