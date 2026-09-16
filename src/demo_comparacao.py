from __future__ import annotations

import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eda_utils as u
import indice_utils as ind

E_SUMULA = re.compile(r"[s5$][uúü][mn]|verbete", re.I)


def avaliar(resolvedor, casos: list[dict]) -> tuple[int, int, int, int]:
    exatas = ambiguas = zero = erradas = 0
    for r in casos:
        candidatos = resolvedor(r["trecho"])
        esperado = int(r["id_canonico"])
        if len(candidatos) == 1 and esperado in candidatos:
            exatas += 1
        elif not candidatos:
            zero += 1
        elif esperado in candidatos:
            ambiguas += 1
        else:
            erradas += 1
    return exatas, ambiguas, zero, erradas


def main() -> None:
    con = u.conectar_base()
    gab = u.carregar_gabarito()
    jur = [r for r in gab if r["tipo"] == "jurisprudencia"]
    reais_jur = [r for r in jur if r["classificacao"] == "real" and not E_SUMULA.search(r["trecho"])]
    inventadas_jur = [r for r in jur if r["classificacao"] == "inventada"]

    print(f"raiz do projeto: {u.raiz_projeto()}")
    print(f"base canônica: {con.execute('SELECT COUNT(*) FROM documentos').fetchone()[0]} registros")
    print(f"gabarito: {len(gab)} citações · testando {len(reais_jur)} reais "
          f"(jurisprudência, sem súmula) e {len(inventadas_jur)} inventadas\n")

    acordaos, sem_parser = ind.construir_indice_acordaos(con)
    print(f"índice de acórdãos: {sum(len(v) for v in acordaos.values())} registros indexados "
          f"({len(sem_parser)} não reconhecidos pelo parser)\n")

    def baseline(trecho: str) -> set[int]:
        if E_SUMULA.search(trecho):
            return set()
        return u.resolver_por_numero(trecho)

    def combinado_resolver(trecho: str) -> set[int]:
        if E_SUMULA.search(trecho):
            return set()
        candidatos = ind.resolver(trecho, acordaos)
        return candidatos if candidatos else u.resolver_por_numero(trecho)

    def combinado_aposta(trecho: str) -> set[int]:
        if E_SUMULA.search(trecho):
            return set()
        candidatos = ind.resolver_com_aposta_de_tamanho(trecho, acordaos)
        return candidatos if candidatos else u.resolver_por_numero(trecho)

    metodos = [
        ("1. Baseline da Camila (sem critério de desempate nosso)", baseline),
        ("2. + ind.resolver (número → classe)", combinado_resolver),
        ("3. + ind.resolver_com_aposta_de_tamanho (número → classe → tamanho)", combinado_aposta),
    ]

    largura = max(len(nome) for nome, _ in metodos)
    print(f"{'método':<{largura}}  exatas        ambíguas  zero  erradas  τ")
    print("-" * (largura + 48))
    for nome, resolvedor in metodos:
        exatas, ambiguas, zero, erradas = avaliar(resolvedor, reais_jur)
        tau = sum(1 for r in inventadas_jur if resolvedor(r["trecho"])) / len(inventadas_jur)
        pct = 100 * exatas / len(reais_jur)
        print(f"{nome:<{largura}}  {exatas}/{len(reais_jur)} ({pct:5.1f}%)  "
              f"{ambiguas:^8}  {zero:^4}  {erradas:^7}  {tau:.3f}")

    print("\n⚠️  O método 3 usa uma heurística (aposta no texto mais longo) validada em"
          " poucos casos do dev set — ver docstring de resolver_com_aposta_de_tamanho"
          " em indice_utils.py antes de considerar usá-la em produção.")


if __name__ == "__main__":
    main()
