"""
Calibração de confiança por caminho de decisão — Desafio 1, BRACIS 2026 / Jusbrasil.

Os valores de `config/pipeline.yaml["confianca"]` eram um chute inicial por
caminho de decisão, sem apoio em dados. Este script substitui o chute por uma
estimativa: roda o pipeline de regras sobre o dev set oficial (26 documentos)
e sobre um corpus sintético maior — gerado por gerar_documentos_sinteticos.py,
com split treino/validação/teste por documento-base —, casa cada predição com
o gabarito pelo mesmo critério oficial (_casar/_iou de data/kaggle_metric.py,
importado direto para não haver deriva da métrica), e para cada caminho estima
a confiança via um posterior Beta-Binomial:

    confianca(caminho) = (acertos + prior·peso) / (n + peso)

`prior` é o valor atual do YAML (preserva a hierarquia de bom senso já
validada pela equipe); `peso` é quantas observações "valem" o prior — com
poucos dados o prior domina, com muitos os dados falam. Caminhos com menos de
`--minimo-n` observações "limpas" não são atualizados: a evidência é
insuficiente (ou, no caso de alguns caminhos de lei/súmula ambígua no corpus
sintético atual, contaminada por ruído de acento Unicode NFD — ver nota
abaixo) para confiar num número novo.

O split "test" do corpus sintético nunca entra no ajuste: serve só para
validar, fora da amostra, que o bônus de Brier realmente melhora com os
valores novos (relatado, não usado para calibrar).

Nota sobre contaminação: um dos modos de ruído de superfície do gerador
sintético (`unicode_nfd`) decompõe acentos (ex. "Súmula" -> "S"+"u"+combining
acute+"m"+"u"+"l"+"a"), o que quebra a extração e o roteamento baseados em
regex — bug real, independente de calibração de confiança (ver ficha de bug
"NFD quebra tokenização"). Citações cujo trecho contém caractere combinante
Unicode são excluídas da calibração para não confundir "caminho pouco
confiável" com "bug do gerador de dados sintéticos".

Duas limitações estruturais do corpus sintético que a exclusão de NFD não
resolve, e que o script trata de forma explícita:

1. Duplicação: os 8 modos de transformação de superfície são determinísticos
   (mesma citação-base + mesmo modo = texto idêntico), então repetir
   `--sintetico-variacoes` além de ~8-24 não acrescenta cenário novo, só
   repete o que já foi visto — o corpus tem ~2x mais linhas do que cenários
   distintos. `_deduplicar` corta isso contando só uma vez por
   (caminho, trecho literal).
2. Circularidade: as citações "reais" extras que o gerador insere
   (`carregar_catalogo` em gerar_documentos_sinteticos.py) são renderizadas a
   partir dos MESMOS índices (`indice_utils.construir_indice_acordaos` /
   `construir_indice_normativos`) que o `Resolvedor` usa para resolver — é
   testar o parser com a sua própria saída, formatada de um jeito limpo que
   uma peça real não necessariamente reproduz. Isso não tem correção simples
   dentro deste script; o efeito prático é que mesmo depois de deduplicar, a
   confiança medida tende a ficar otimista. Por isso `--teto-confianca`
   (padrão 0,98) trava o valor calibrado abaixo de 1,0 mesmo quando o corpus
   mostra acerto "perfeito" — é um voto de desconfiança deliberado na
   generalização para o conjunto oculto, que a contagem sozinha não capta.

Os caminhos de súmula, `acordao_fts` e, em bases antigas, os de tema
nunca acumulam `--minimo-n` de forma confiável mesmo com corpus sintético
grande, por razões estruturais que mais dado sintético não resolve:
`acordao_fts` só existe quando o parser de cabeçalho falha — e o catálogo do
gerador sintético é construído a partir do índice já parseado, então NUNCA
inclui um registro sem cabeçalho parseável; `sumula_real`/`sumula_inventada`
competem por um sorteio aleatório num universo de só 5 súmulas contra ~1000
acórdãos. Bases canônicas que contenham registros com `natureza='tema'`
também alimentam o corpus sintético; bases antigas, sem esses registros, não
oferecem evidência para calibrar os caminhos de tema. Os valores no
YAML para esses caminhos foram ajustados manualmente a partir de evidência
real (não do corpus sintético circular), registrada como teste permanente em
tests/test_cenarios.py:
  - `sumula_real`/`sumula_inventada`: as 5 súmulas da base inteira (universo
    completo, não amostra) mais 5 números inventados plausíveis — todas
    corretas; aplicado o mesmo encolhimento Beta-Binomial (peso=20) sobre
    n=5.
  - `acordao_fts`: 2 casos reais construídos a partir de acórdãos que
    genuinamente não têm cabeçalho parseável (`sem_parser` de
    `construir_indice_acordaos`) — 1 resolvido corretamente via FTS, 1
    conhecido como falha (`test_fts_nao_resolve_acordao_com_encoding_
    severamente_corrompido`, `@unittest.expectedFailure`: corrupção de
    encoding severa demais para o FTS indexar o número). O valor caiu de
    0,90 para 0,86 — a evidência real aponta pra esse caminho ser MENOS
    confiável do que o chute original assumia, não mais.
  - caminhos de tema: mantêm valores conservadores até haver amostra
    suficiente da nova base. Um tema agora pode ser `real`, `inventada` ou
    `incompleta`, conforme a consulta ao índice canônico.

Os 7 caminhos restantes (`lei_sem_artigo`, `lei_sem_estatuto`,
`sumula_sem_numero`, `sumula_sem_tribunal`, `lei_ambigua`, `sumula_ambigua`,
`acordao_ambigua`) nunca ocorrem nos 26 documentos de dev nem de forma
confiável no corpus sintético — eram os únicos ainda no chute inicial do
projeto (0,50/0,60). Revisão manual (01/10/2026): esse chute estava baixo
demais frente à capacidade real do pipeline nos demais caminhos, mas os 7 não
são todos igualmente arriscados, então o ajuste foi diferenciado, não
uniforme:
  - `lei_sem_artigo`/`lei_sem_estatuto`/`sumula_sem_numero`/
    `sumula_sem_tribunal` -> 0,85. São casos de campo ausente: a classificação
    "incompleta" é quase tautológica — se o artigo/diploma/número/tribunal
    não foi encontrado, a citação é mesmo incompleta por definição. A única
    forma de errar é a extração/roteamento terem pegado o span errado, não
    uma falha deste caminho em si; por isso o valor fica perto dos outros
    caminhos "logicamente garantidos" (tema, lei_inventada).
  - `lei_ambigua`/`sumula_ambigua`/`acordao_ambigua` -> 0,80. São colisão
    genuína entre candidatos — mais arriscado que "campo ausente", porque
    sempre existe a possibilidade de uma heurística de desempate melhor
    resolver o que hoje tratamos como ambíguo (ver a investigação de
    acordao_ambigua: o único exemplo real que testamos só parecia ambíguo
    por causa de um bug do gerador sintético, não por falha do pipeline —
    mas isso não é prova de que NUNCA existe desempate possível). Por isso
    fica um degrau abaixo do primeiro grupo.

Uso:
    python src/calibrar_confianca.py                  # relatório, não escreve nada
    python src/calibrar_confianca.py --aplicar         # atualiza config/pipeline.yaml
    python src/calibrar_confianca.py --relatorio saida/calibracao.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import eda_utils as u
from configuracao import PASTA_CONFIG, carregar_config
from steps.buscar_na_base import Resolvedor
from steps.extrair_candidatos import extrair
from steps.filtrar_distratores import filtrar_distratores
from steps.normalizar_citacao import normalizar_citacao
from steps.pipeline_utils import DADOS, RAIZ, carregar_modulo
from steps.rotear_citacao import rotear

CONFIG_PIPELINE = PASTA_CONFIG / "pipeline.yaml"
KAGGLE_METRIC = carregar_modulo(DADOS / "kaggle_metric.py", "kaggle_metric")


# --------------------------------------------------------------------------- #
# Coleta: roda o pipeline e casa com o gabarito pelo critério oficial
# --------------------------------------------------------------------------- #


def _norm_id(v: str | None) -> str | None:
    v = (v or "").strip()
    if not v:
        return None
    return v.lstrip("0") or "0" if v.isdigit() else v


def _tem_combinante(s: str) -> bool:
    return any(unicodedata.combining(c) for c in s)


def carregar_goldenset_generico(caminho_csv: Path) -> dict[str, list[dict]]:
    """Lê um goldenset_offsets.csv (dev real ou sintético — mesmo schema)."""
    with caminho_csv.open(encoding="utf-8-sig", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    golds_por_doc: dict[str, list[dict]] = defaultdict(list)
    for linha in linhas:
        classe = linha["classificacao"]
        doc_ids = frozenset([_norm_id(linha["id_canonico"])]) if classe == "real" else frozenset()
        golds_por_doc[linha["documento_id"]].append({
            "inicio": int(linha["inicio"]), "fim": int(linha["fim"]),
            "classe": classe, "doc_ids": doc_ids, "nivel": int(linha["nivel"]),
        })
    return dict(golds_por_doc)


def processar_corpus(pasta_txt: Path, goldenset_csv: Path, resolvedor: Resolvedor) -> list[dict]:
    """Roda o pipeline de regras (etapas 1-5) e casa com o gabarito.

    Devolve um registro por par casado: caminho de decisão, se o par está
    contaminado por ruído Unicode NFD, e ``y`` — 1 se a classe bateu e (para
    ``real``) o id_canonico também, exatamente a definição usada pelo termo
    de Brier em kaggle_metric.py.
    """
    golds_por_doc = carregar_goldenset_generico(goldenset_csv)
    registros: list[dict] = []
    # Recursivo: o dev set é plano (data/txt/*.txt), o corpus sintético usa
    # subpastas por split (txt/train/*.txt, txt/validation/*.txt, txt/test/*.txt).
    for arquivo in sorted(pasta_txt.glob("**/*.txt")):
        doc_id = arquivo.stem
        golds = golds_por_doc.get(doc_id)
        if not golds:
            continue
        texto = arquivo.read_bytes().decode("utf-8")
        candidatos = filtrar_distratores(extrair(texto), texto)
        preds: list[dict] = []
        extra: dict[tuple[int, int], dict] = {}
        for candidato in candidatos:
            rota = rotear(candidato)
            candidato_norm = dict(candidato)
            candidato_norm["_contexto_esquerda"] = texto[max(0, candidato["inicio"] - 48):candidato["inicio"]]
            candidato_norm["_contexto_direita"] = texto[candidato["fim"]:candidato["fim"] + 48]
            citacao = normalizar_citacao(candidato_norm, rota)
            resultado = resolvedor.resolver_citacao(citacao)
            id_str = str(resultado.id_canonico) if resultado.id_canonico is not None else None
            preds.append({"inicio": candidato["inicio"], "fim": candidato["fim"],
                          "classe": resultado.classe, "id_canonico": id_str})
            janela = texto[max(0, candidato["inicio"] - 3):candidato["fim"] + 3]
            extra[(candidato["inicio"], candidato["fim"])] = {
                "caminho": resultado.caminho, "contaminado": _tem_combinante(janela),
                "trecho": candidato["trecho"],
            }
        pares, _, _ = KAGGLE_METRIC._casar(golds, preds)
        for gi, pi in pares:
            g, pr = golds[gi], preds[pi]
            acertou_classe = g["classe"] == pr["classe"]
            y = 1 if acertou_classe and (g["classe"] != "real" or pr["id_canonico"] in g["doc_ids"]) else 0
            info = extra[(pr["inicio"], pr["fim"])]
            registros.append({
                "caminho": info["caminho"], "contaminado": info["contaminado"],
                "trecho": info["trecho"], "y": y, "nivel": g["nivel"], "documento_id": doc_id,
            })
    return registros


def preparar_corpus_sintetico(diretorio: Path, txt_dev: Path, goldenset_dev: Path, db: Path,
                              variacoes: int, seed: int) -> Path:
    """Gera (ou reaproveita, se já existir) o corpus sintético de calibração."""
    if not (diretorio / "goldenset_offsets.csv").exists():
        import gerar_documentos_sinteticos as gerador
        diretorio.mkdir(parents=True, exist_ok=True)
        gerador.executar(txt_dev, goldenset_dev, db, diretorio, variacoes, seed)
    return diretorio


# --------------------------------------------------------------------------- #
# Estimativa: posterior Beta-Binomial com encolhimento para o prior atual
# --------------------------------------------------------------------------- #


def confianca_calibrada(acertos: int, n: int, prior: float, peso: float) -> float:
    a = prior * peso
    b = (1.0 - prior) * peso
    return (acertos + a) / (n + a + b)


def _deduplicar(registros: list[dict]) -> list[dict]:
    """Mantém só uma observação por (caminho, trecho literal).

    O gerador sintético aplica transformações determinísticas: a mesma receita
    (mesma citação-base + mesmo modo de ruído) produz texto idêntico toda vez
    que se repete entre variações. Contar cada repetição como uma observação
    nova infla artificialmente o `n` e, com ele, a confiança no dado — o
    corpus tem ~2x mais linhas do que cenários realmente distintos. Isso não
    resolve a circularidade de as citações "extras" serem geradas a partir do
    índice canônico (ver nota no docstring do módulo), só a duplicata exata.
    """
    vistos: set[tuple[str, str]] = set()
    unicos = []
    for r in registros:
        chave = (r["caminho"], r["trecho"])
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(r)
    return unicos


def calibrar(registros_ajuste: list[dict], prior_atual: dict[str, float],
            peso: float, minimo_n: int, teto: float) -> dict[str, dict]:
    limpos = _deduplicar([r for r in registros_ajuste if not r["contaminado"]])
    por_caminho: dict[str, list[dict]] = defaultdict(list)
    for r in limpos:
        por_caminho[r["caminho"]].append(r)

    resultado = {}
    for caminho, prior in prior_atual.items():
        itens = por_caminho.get(caminho, [])
        n = len(itens)
        acertos = sum(r["y"] for r in itens)
        bruto = confianca_calibrada(acertos, n, prior, peso) if n >= minimo_n else prior
        # Teto deliberado abaixo de 1,0: mesmo com acerto "perfeito" no corpus
        # de calibração, não generalizamos cegamente para o conjunto oculto —
        # o corpus sintético deriva de só 26 documentos e uma parte das
        # citações "reais" é gerada a partir do próprio índice canônico que o
        # resolvedor usa (ver nota no docstring do módulo). O teto expressa
        # essa incerteza estrutural que a contagem sozinha não capta.
        novo = min(bruto, teto)
        resultado[caminho] = {
            "n": n,
            "acertos": acertos,
            "acc_bruta": round(acertos / n, 4) if n else None,
            "valor_atual": prior,
            "valor_novo": round(novo, 3),
            "atualizado": n >= minimo_n,
        }
    return resultado


def brier_bonus(registros: list[dict], valores: dict[str, float]) -> tuple[float, float] | tuple[None, None]:
    limpos = [r for r in registros if not r["contaminado"] and r["caminho"] in valores]
    if not limpos:
        return None, None
    termos = [(valores[r["caminho"]] - r["y"]) ** 2 for r in limpos]
    brier = sum(termos) / len(termos)
    return brier, max(0.0, min(0.10, 0.10 * (1.0 - brier)))


# --------------------------------------------------------------------------- #
# Relatório e escrita no YAML
# --------------------------------------------------------------------------- #


def imprimir_relatorio(estimativas: dict[str, dict], validacao: dict) -> None:
    print(f"{'caminho':<20} {'n':>6} {'acc_bruta':>10} {'atual':>7} {'novo':>7} {'':>10}")
    for caminho, info in sorted(estimativas.items(), key=lambda kv: -kv[1]["n"]):
        marca = "" if info["atualizado"] else "  (mantido: n insuficiente)"
        acc = f"{info['acc_bruta']:.4f}" if info["acc_bruta"] is not None else "-"
        print(f"{caminho:<20} {info['n']:>6} {acc:>10} {info['valor_atual']:>7.3f} "
              f"{info['valor_novo']:>7.3f} {marca}")
    if validacao.get("brier_antigo") is not None:
        print(f"\nvalidação fora da amostra (split de teste sintético, n={validacao['n']}):")
        print(f"  brier com valores atuais : {validacao['brier_antigo']:.4f} "
              f"-> bônus {validacao['bonus_antigo']:.4f}")
        print(f"  brier com valores novos  : {validacao['brier_novo']:.4f} "
              f"-> bônus {validacao['bonus_novo']:.4f}")


def _formatar_valor(v: float) -> str:
    s = f"{v:.3f}".rstrip("0")
    return s if not s.endswith(".") else s + "0"


def atualizar_yaml(caminho_yaml: Path, estimativas: dict[str, dict]) -> int:
    """Substitui só os valores que mudaram, preservando o resto do arquivo."""
    texto = caminho_yaml.read_text(encoding="utf-8")
    atualizados = 0
    for caminho, info in estimativas.items():
        if not info["atualizado"] or abs(info["valor_novo"] - info["valor_atual"]) < 1e-9:
            continue
        padrao = re.compile(rf'("{re.escape(caminho)}"\s*:\s*)[0-9]*\.?[0-9]+')
        novo_texto, n = padrao.subn(
            lambda m: m.group(1) + _formatar_valor(info["valor_novo"]), texto, count=1,
        )
        if n != 1:
            raise ValueError(f"chave de confiança {caminho!r} não encontrada em {caminho_yaml}")
        texto = novo_texto
        atualizados += 1
    caminho_yaml.write_text(texto, encoding="utf-8")
    return atualizados


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--txt", type=Path, default=DADOS / "txt")
    parser.add_argument("--goldenset", type=Path, default=DADOS / "goldenset_offsets.csv")
    parser.add_argument("--db", type=Path, default=DADOS / "desafio1_bracis.db")
    parser.add_argument("--config", type=Path, default=CONFIG_PIPELINE)
    parser.add_argument("--sintetico-dir", type=Path, default=RAIZ / "saida" / "sintetico_calibracao",
                        help="cache do corpus sintético; gerado se não existir")
    parser.add_argument("--sintetico-variacoes", type=int, default=40)
    parser.add_argument("--sintetico-seed", type=int, default=20260930)
    parser.add_argument("--sem-sintetico", action="store_true",
                        help="calibra só com o dev set (não recomendado: satura em 100%% de acerto)")
    parser.add_argument("--peso-prior", type=float, default=20.0,
                        help="pseudo-contagens do prior no posterior Beta-Binomial")
    parser.add_argument("--minimo-n", type=int, default=30,
                        help="abaixo disso (após deduplicar), mantém o valor atual (evidência insuficiente)")
    parser.add_argument("--teto-confianca", type=float, default=0.98,
                        help="nenhum valor calibrado passa disso, por mais perfeito que o corpus pareça "
                             "(hedge contra generalização além da amostra de 26 documentos-base)")
    parser.add_argument("--aplicar", action="store_true", help="escreve os valores novos no YAML")
    parser.add_argument("--relatorio", type=Path, help="grava o relatório completo em JSON")
    args = parser.parse_args()

    con = sqlite3.connect(u._uri_sqlite(args.db.resolve()) + "?immutable=1", uri=True)
    resolvedor = Resolvedor(con)

    print(f"processando dev set ({args.txt})...")
    registros = processar_corpus(args.txt, args.goldenset, resolvedor)
    registros_teste: list[dict] = []

    if not args.sem_sintetico:
        print(f"preparando corpus sintético ({args.sintetico_dir})...")
        sintetico = preparar_corpus_sintetico(
            args.sintetico_dir, args.txt, args.goldenset, args.db,
            args.sintetico_variacoes, args.sintetico_seed,
        )
        print("processando corpus sintético (pode levar alguns minutos)...")
        goldenset_sint = sintetico / "goldenset_offsets.csv"
        golds_por_doc = carregar_goldenset_generico(goldenset_sint)
        splits = {doc_id: doc_id.split("_")[1] for doc_id in golds_por_doc}
        todos = processar_corpus(sintetico, goldenset_sint, resolvedor)
        for r in todos:
            if splits.get(r["documento_id"]) == "test":
                registros_teste.append(r)
            else:
                registros.append(r)

    prior_atual = carregar_config("pipeline")["confianca"]
    estimativas = calibrar(registros, prior_atual, args.peso_prior, args.minimo_n, args.teto_confianca)

    validacao: dict = {}
    if registros_teste:
        valores_novos = {c: info["valor_novo"] for c, info in estimativas.items()}
        brier_antigo, bonus_antigo = brier_bonus(registros_teste, prior_atual)
        brier_novo, bonus_novo = brier_bonus(registros_teste, valores_novos)
        validacao = {
            "n": sum(1 for r in registros_teste if not r["contaminado"]),
            "brier_antigo": brier_antigo, "bonus_antigo": bonus_antigo,
            "brier_novo": brier_novo, "bonus_novo": bonus_novo,
        }

    print()
    imprimir_relatorio(estimativas, validacao)

    if args.relatorio:
        args.relatorio.parent.mkdir(parents=True, exist_ok=True)
        args.relatorio.write_text(
            json.dumps({"estimativas": estimativas, "validacao": validacao}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nrelatório: {args.relatorio}")

    if args.aplicar:
        n = atualizar_yaml(args.config, estimativas)
        print(f"\n{args.config}: {n} valores atualizados.")
    else:
        print("\n(modo relatório — rode com --aplicar para escrever no YAML)")


if __name__ == "__main__":
    main()
