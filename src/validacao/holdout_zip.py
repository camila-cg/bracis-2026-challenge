"""Avalia o corpus sintético versionado sem alterar o ZIP original.

O dataset atual em ``datasets/`` contém 26 documentos e 71 anotações. Este
módulo gera uma cópia compatível com a métrica oficial, executa o pipeline e
produz duas leituras: o score oficial e um diagnóstico de FN, FP, classe e ID.
Os 26 documentos também constam no treino do detector final. Portanto, este
gate mede regressões das regras e do contrato de avaliação, mas não certifica
generalização do modelo ou da fusão híbrida. Para isso, use ``gate_hibrido.py``
com um corpus independente e manifestos de proveniência.

Há duas camadas diferentes, que não devem ser confundidas:

1. **Compatibilidade técnica (sempre aplicada, sem mudar a semântica)**

   - 15 citações ``real`` usam ``id_canonico=doc_####``; o contrato oficial
     exige o ID numérico. O mapeamento é aceito somente quando ``documento_id``
     identifica exatamente um registro na base.
   - Os documentos ``jbsg26-<nível>-<n>-<hash>`` são renomeados para
     ``gen_n<nível>_<n>``, formato lido por :func:`eda_utils.nivel`.

2. **Curadoria de política (padrão do gate, altera o gabarito)**

   - Quatro anotações ``lei/incompleta`` sem artigo são removidas. Essa decisão
     acompanha a fronteira adotada no gabarito oficial revisado, que deixou de
     anotar alusões sem fonte concreta. Porém, ela diverge da política declarada
     em ``generation_parameters.json`` do próprio ZIP, que permite lei
     identificável sem dispositivo. Portanto, é uma escolha do projeto, não uma
     correção factual do dataset.
   - Cinco anotações ``real`` apontam para acórdãos duplicados, com a mesma
     identidade consultável. Como o texto não distingue qual ID duplicado seria
     o alvo, elas viram ``incompleta`` e perdem o ID. Isso evita premiar uma
     escolha arbitrária, mas também é uma decisão metodológica.

Consequência: o score curated não é o desempenho objetivo no gabarito bruto e
deve ser apresentado junto dessas decisões. No modo curated, 71 anotações viram
67; cinco das 67 mudam de ``real`` para ``incompleta``. O modo raw preserva as
71 anotações e suas classes, aplicando apenas a compatibilidade técnica.

Uso, a partir da raiz do repositório::

    # Curated: modo usado pelo gate de regressão.
    python src/validacao/holdout_zip.py

    # Raw: preserva as anotações e classes do ZIP.
    python src/validacao/holdout_zip.py --manter-difusas --manter-duplicatas

    # Lista os erros aproximados encontrados em qualquer modo.
    python src/validacao/holdout_zip.py --detalhes

O ZIP nunca é regravado. Textos, CSV adaptado e saída são escritos apenas no
destino informado; nos testes, esse destino é temporário.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sqlite3
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eda_utils as u
import pipeline as p
from steps.avaliar_metrica import avaliar, imprimir_avaliacao
from steps.salvar_submissao import gravar_saida
from validacao.teste_base_inteira import ids_duplicados

ZIP = RAIZ / "datasets" / "dataset_citacoes_br_sintetico_2026_seed26092217.zip"
_PASTA_NO_ZIP = "dataset_citacoes_br_sintetico_2026/"
_NOME_DOCUMENTO = re.compile(r"^jbsg26-(?P<nivel>[12])-(?P<seq>\d{2})-[0-9a-f]+$")
_ARTIGO_NUMERADO = re.compile(r"\bart(?:igo)?\.?\s*\d", re.I)
COLUNAS = ["nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho", "tipo", "classificacao", "id_canonico"]


@dataclass
class Saneado:
    textos: dict[str, str]
    gabarito: list[dict]
    relatorio: dict


def renomear_documento(documento_id: str) -> str:
    """jbsg26-2-14-<hash> -> gen_n2_014, o formato que eda_utils.nivel() lê."""
    achado = _NOME_DOCUMENTO.match(documento_id)
    if achado is None:
        raise ValueError(f"nome de documento fora do padrão do ZIP: {documento_id}")
    return f"gen_n{achado.group('nivel')}_{int(achado.group('seq')):03d}"


def _e_alusao_sem_artigo(linha: dict) -> bool:
    return (
        linha["tipo"] == "lei"
        and linha["classificacao"] == "incompleta"
        and not _ARTIGO_NUMERADO.search(linha["trecho"])
    )


def sanear(
    caminho_zip: Path,
    con: sqlite3.Connection,
    manter_difusas: bool = False,
    duplicados: set[int] | None = None,
) -> Saneado:
    """Cria a representação compatível e, opcionalmente, aplica a curadoria.

    A renomeação dos documentos e o remapeamento ``doc_####`` → ID numérico são
    sempre aplicados, pois apenas adaptam o contrato técnico. ``manter_difusas``
    preserva as quatro leis incompletas do gerador. ``duplicados`` contém IDs
    indistinguíveis que serão reclassificados para ``incompleta``; ``None``
    preserva as classes originais. Assim, ``manter_difusas=True`` e
    ``duplicados=None`` correspondem ao modo raw documentado no módulo.

    Retorna textos e gabarito em memória, além de um relatório que explicita
    quantos IDs foram adaptados e quais anotações sofreram curadoria.
    """
    with zipfile.ZipFile(caminho_zip) as arquivo:
        bruto = arquivo.read(_PASTA_NO_ZIP + "goldenset_offsets.csv").decode("utf-8-sig")
        textos_originais = {
            Path(nome).stem: arquivo.read(nome).decode("utf-8")
            for nome in arquivo.namelist()
            if nome.endswith(".txt")
        }
    linhas = list(csv.DictReader(io.StringIO(bruto)))

    nomes = {documento: renomear_documento(documento) for documento in textos_originais}
    if len(set(nomes.values())) != len(nomes):
        raise ValueError("renomear_documento gerou nomes repetidos")
    textos = {nomes[documento]: texto for documento, texto in textos_originais.items()}

    gabarito, removidas, reclassificadas, remapeados = [], [], [], 0
    for linha in linhas:
        if not manter_difusas and _e_alusao_sem_artigo(linha):
            removidas.append(linha["trecho"])
            continue
        novo = dict(linha)
        novo["documento_id"] = nomes[linha["documento_id"]]
        if linha["classificacao"] == "real" and not linha["id_canonico"].isdigit():
            achados = con.execute(
                "SELECT id FROM documentos WHERE documento_id = ?", (linha["id_canonico"],)
            ).fetchall()
            if len(achados) != 1:
                raise ValueError(f"id_canonico {linha['id_canonico']!r} não mapeia para um único id")
            novo["id_canonico"] = str(achados[0][0])
            remapeados += 1
        if duplicados and novo["classificacao"] == "real" and int(novo["id_canonico"]) in duplicados:
            reclassificadas.append(linha["trecho"])
            novo["classificacao"], novo["id_canonico"] = "incompleta", ""
        gabarito.append(novo)

    relatorio = {
        "documentos": len(textos),
        "citacoes_originais": len(linhas),
        "citacoes_saneadas": len(gabarito),
        "ids_remapeados": remapeados,
        "alusoes_removidas": removidas,
        "reclassificadas_por_duplicata": reclassificadas,
    }
    return Saneado(textos, gabarito, relatorio)


def gravar(saneado: Saneado, destino: Path) -> Path:
    """Escreve `txt/` e `goldenset_offsets.csv` em `destino`; devolve o caminho do CSV."""
    (destino / "txt").mkdir(parents=True, exist_ok=True)
    for documento, texto in saneado.textos.items():
        with (destino / "txt" / f"{documento}.txt").open("w", encoding="utf-8", newline="") as arquivo:
            arquivo.write(texto)
    gabarito = destino / "goldenset_offsets.csv"
    with gabarito.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(saneado.gabarito)
    return gabarito


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    uniao = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / uniao if uniao else 0.0


def comparar(documentos: list[dict], gabarito: list[dict]) -> dict[str, list]:
    """Diagnóstico aproximado por pareamento guloso com IoU >= 0,5 (a métrica oficial manda no score)."""
    resultado = {"pareadas": [], "falsos_negativos": [], "falsos_positivos": [],
                 "classe_errada": [], "id_errado": []}
    por_documento: dict[str, list[dict]] = {}
    for linha in gabarito:
        por_documento.setdefault(linha["documento_id"], []).append(linha)
    for documento in documentos:
        esperadas = por_documento.get(documento["documento_id"], [])
        livres = list(range(len(esperadas)))
        for predita in documento["citacoes"]:
            melhor, melhor_iou = None, 0.0
            for i in livres:
                iou = _iou((predita["inicio"], predita["fim"]),
                           (int(esperadas[i]["inicio"]), int(esperadas[i]["fim"])))
                if iou > melhor_iou:
                    melhor, melhor_iou = i, iou
            if melhor is None or melhor_iou < 0.5:
                resultado["falsos_positivos"].append((documento["documento_id"], predita["trecho"], predita["classificacao"]))
                continue
            livres.remove(melhor)
            esperada = esperadas[melhor]
            resultado["pareadas"].append((documento["documento_id"], predita["trecho"]))
            if predita["classificacao"] != esperada["classificacao"]:
                resultado["classe_errada"].append((
                    documento["documento_id"], predita["trecho"],
                    esperada["classificacao"], predita["classificacao"]))
            elif esperada["classificacao"] == "real":
                obtido = (predita["resolucao"] or {}).get("id_canonico")
                if obtido != esperada["id_canonico"]:
                    resultado["id_errado"].append((
                        documento["documento_id"], predita["trecho"], esperada["id_canonico"], obtido))
        for i in livres:
            resultado["falsos_negativos"].append((
                documento["documento_id"], esperadas[i]["trecho"], esperadas[i]["classificacao"]))
    return resultado


def executar(
    destino: Path, manter_difusas: bool = False, manter_duplicatas: bool = False
) -> tuple[dict, dict, Saneado]:
    """Adapta o holdout, executa o pipeline e calcula suas duas avaliações.

    Por padrão aplica a curadoria de alusões e duplicatas usada pelo gate.
    Quando ambos os parâmetros ``manter_*`` são verdadeiros, preserva o
    gabarito raw e aplica somente as adaptações técnicas inevitáveis.

    Retorna ``(métrica_oficial, diagnóstico_aproximado, dados_adaptados)``.
    O diagnóstico usa pareamento guloso por IoU apenas para explicar erros; o
    score que define o gate é sempre calculado pela métrica oficial.
    """
    con = u.conectar_base()
    resolvedor = p.Resolvedor(con)
    duplicados = None if manter_duplicatas else ids_duplicados(resolvedor)
    saneado = sanear(ZIP, con, manter_difusas, duplicados)
    gabarito = gravar(saneado, destino)
    documentos = [
        p.processar_documento(documento, texto, resolvedor)
        for documento, texto in sorted(saneado.textos.items())
    ]
    submissao = gravar_saida(documentos, destino / "saida")
    return avaliar(submissao, gabarito), comparar(documentos, saneado.gabarito), saneado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destino", type=Path, default=RAIZ / "saida" / "holdout_zip")
    parser.add_argument("--manter-difusas", action="store_true",
                        help="não remove as alusões de lei sem número de artigo")
    parser.add_argument("--manter-duplicatas", action="store_true",
                        help="não reclassifica como incompleta as reais que apontam para acórdãos duplicados")
    parser.add_argument("--detalhes", action="store_true", help="lista cada erro")
    args = parser.parse_args()

    resultado, diagnostico, saneado = executar(args.destino, args.manter_difusas, args.manter_duplicatas)
    rel = saneado.relatorio
    print(f"\nSaneamento: {rel['citacoes_originais']} -> {rel['citacoes_saneadas']} citações em "
          f"{rel['documentos']} documentos; {rel['ids_remapeados']} ids remapeados; "
          f"{len(rel['alusoes_removidas'])} alusões removidas; "
          f"{len(rel['reclassificadas_por_duplicata'])} reais reclassificadas por duplicata.")
    imprimir_avaliacao(resultado)
    print(f"\nDiagnóstico aproximado (IoU >= 0,5): {len(diagnostico['pareadas'])} pareadas, "
          f"{len(diagnostico['falsos_negativos'])} falsos negativos, "
          f"{len(diagnostico['falsos_positivos'])} falsos positivos, "
          f"{len(diagnostico['classe_errada'])} com classe errada, "
          f"{len(diagnostico['id_errado'])} com id errado.")
    if args.detalhes:
        for nome, itens in diagnostico.items():
            if nome == "pareadas" or not itens:
                continue
            print(f"\n{nome}:")
            for item in itens:
                print("  ", " | ".join(str(campo).replace("\n", " ")[:70] for campo in item))


if __name__ == "__main__":
    main()
