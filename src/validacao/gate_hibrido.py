"""Gate offline e pareado para regras, detector BIO e fusão híbrida.

O gate não escolhe limiares. Os dois valores são entradas obrigatórias e a
avaliação é recusada quando o corpus compartilha IDs ou bytes com o treino.
Cada texto passa pelo detector exatamente uma vez; as mesmas predições são
reutilizadas nas ablações ``modelo`` e ``fusao``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import eda_utils as u
import pipeline as p
from ml_citacoes.fusao import fusionar
from steps.buscar_na_base import Resolvedor
from steps.extrair_candidatos import extrair
from steps.filtrar_distratores import e_distrator, filtrar_distratores
from steps.salvar_submissao import gravar_saida
from validacao.manifesto_datasets import (
    ErroManifesto,
    auditar_sobreposicao,
    carregar_manifesto,
    garantir_sem_vazamento,
)

MODOS = ("regras", "modelo", "fusao")
TOLERANCIA = 1e-12
VERSAO_RELATORIO = 1


class ErroGate(RuntimeError):
    """Falha de integridade, execução ou qualidade do gate."""

    def __init__(self, mensagem: str, *, relatorio: dict[str, Any] | None = None):
        super().__init__(mensagem)
        self.relatorio = relatorio


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _sha256_artefato(caminho: Path) -> str:
    """Hash de arquivo ou árvore, incluindo nomes relativos e conteúdo."""
    if caminho.is_file():
        return _sha256(caminho)
    if not caminho.is_dir():
        raise ErroGate(f"artefato inexistente: {caminho}")
    digest = hashlib.sha256()
    for arquivo in sorted(
        (item for item in caminho.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(caminho).as_posix(),
    ):
        relativo = arquivo.relative_to(caminho).as_posix()
        digest.update(relativo.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(arquivo.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(arquivo).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _commit_git() -> str | None:
    try:
        processo = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RAIZ,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = processo.stdout.strip()
    return commit if len(commit) == 40 else None


def _worktree_limpa() -> bool | None:
    try:
        processo = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=RAIZ,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return not processo.stdout.strip()


def _validar_limiar(nome: str, valor: float) -> float:
    if isinstance(valor, bool) or not isinstance(valor, Real):
        raise ErroGate(f"{nome} deve ser um número entre 0 e 1")
    normalizado = float(valor)
    if not math.isfinite(normalizado) or not 0.0 <= normalizado <= 1.0:
        raise ErroGate(f"{nome} deve estar entre 0 e 1")
    return normalizado


def _documentos_manifestados(manifesto: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    documentos: dict[str, dict[str, Any]] = {}
    for dataset in manifesto.get("datasets", []):
        for documento in dataset.get("documentos", []):
            documento_id = documento["documento_id"]
            if documento_id in documentos:
                raise ErroGate(
                    f"documento {documento_id!r} aparece mais de uma vez no manifesto de avaliação"
                )
            documentos[documento_id] = documento
    if not documentos:
        raise ErroGate("manifesto de avaliação sem documentos")
    return documentos


def _carregar_textos(pasta: Path, manifesto: Mapping[str, Any]) -> dict[str, str]:
    if not pasta.is_dir():
        raise ErroGate(f"pasta de textos inexistente: {pasta}")
    arquivos = sorted(pasta.glob("*.txt"), key=lambda item: item.name)
    por_id = {arquivo.stem: arquivo for arquivo in arquivos}
    if len(por_id) != len(arquivos):
        raise ErroGate("documento_id duplicado na pasta de textos")

    esperados = _documentos_manifestados(manifesto)
    if set(por_id) != set(esperados):
        faltantes = sorted(set(esperados) - set(por_id))
        extras = sorted(set(por_id) - set(esperados))
        raise ErroGate(
            f"textos divergem do manifesto: faltantes={faltantes[:5]}, extras={extras[:5]}"
        )

    textos: dict[str, str] = {}
    for documento_id in sorted(por_id):
        arquivo = por_id[documento_id]
        esperado = esperados[documento_id]
        if _sha256(arquivo) != esperado["sha256"]:
            raise ErroGate(f"SHA-256 do texto diverge do manifesto: {documento_id}")
        if arquivo.stat().st_size != esperado["tamanho_bytes"]:
            raise ErroGate(f"tamanho do texto diverge do manifesto: {documento_id}")
        textos[documento_id] = p.ler_texto(arquivo)
    return textos


def _normalizar_json(valor: Any) -> Any:
    """Converte resultados do scorer (inclusive escalares NumPy) em JSON estável."""
    if isinstance(valor, Mapping):
        return {
            str(chave): _normalizar_json(item)
            for chave, item in sorted(valor.items(), key=lambda par: str(par[0]))
        }
    if isinstance(valor, (list, tuple)):
        return [_normalizar_json(item) for item in valor]
    if isinstance(valor, bool) or valor is None or isinstance(valor, str):
        return valor
    if isinstance(valor, Integral):
        return int(valor)
    if isinstance(valor, Real):
        numero = float(valor)
        if not math.isfinite(numero):
            raise ErroGate("relatório contém número não finito")
        return numero
    if hasattr(valor, "item"):
        return _normalizar_json(valor.item())
    raise ErroGate(f"valor não serializável no relatório: {type(valor).__name__}")


def _gravar_json_deterministico(conteudo: Mapping[str, Any], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(_normalizar_json(conteudo), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _metricas_niveis(metricas: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    try:
        score_final = float(metricas["score_final"])
        niveis = metricas["niveis"]
    except (KeyError, TypeError, ValueError) as erro:
        raise ErroGate("scorer não retornou score_final e niveis válidos") from erro
    if not math.isfinite(score_final) or not isinstance(niveis, Mapping) or not niveis:
        raise ErroGate("scorer retornou métricas inválidas")
    normalizados = {str(nivel): resultado for nivel, resultado in niveis.items()}
    if set(normalizados) != {"1", "2"}:
        raise ErroGate(
            f"scorer deve retornar exatamente os níveis 1 e 2: {sorted(normalizados)}"
        )
    for nivel, resultado in normalizados.items():
        if not isinstance(resultado, Mapping):
            raise ErroGate(f"métricas inválidas para o nível {nivel}")
        faltantes = {"score", "macro_f1", "f1_por_classe", "tau", "s", "b"} - set(resultado)
        if faltantes:
            raise ErroGate(f"métricas incompletas no nível {nivel}: {sorted(faltantes)}")
        classes = resultado["f1_por_classe"]
        if not isinstance(classes, Mapping):
            raise ErroGate(f"f1_por_classe inválido no nível {nivel}")
        classes_faltantes = {"real", "inventada", "incompleta"} - set(classes)
        if classes_faltantes:
            raise ErroGate(
                f"F1 por classe incompleto no nível {nivel}: {sorted(classes_faltantes)}"
            )
    return normalizados


def _checagens_nao_regressao(
    regras: Mapping[str, Any], fusao: Mapping[str, Any]
) -> list[dict[str, Any]]:
    niveis_regras = _metricas_niveis(regras)
    niveis_fusao = _metricas_niveis(fusao)
    if set(niveis_regras) != set(niveis_fusao):
        raise ErroGate("regras e fusão foram avaliadas em níveis diferentes")

    checagens = []

    def maior_ou_igual(nome: str, obtido: float, referencia: float) -> None:
        checagens.append(
            {
                "criterio": nome,
                "obtido": obtido,
                "referencia": referencia,
                "aprovado": obtido + TOLERANCIA >= referencia,
            }
        )

    def menor_ou_igual(nome: str, obtido: float, referencia: float) -> None:
        checagens.append(
            {
                "criterio": nome,
                "obtido": obtido,
                "referencia": referencia,
                "aprovado": obtido <= referencia + TOLERANCIA,
            }
        )

    maior_ou_igual(
        "fusao.score_final >= regras.score_final",
        float(fusao["score_final"]),
        float(regras["score_final"]),
    )
    for nivel in sorted(niveis_regras):
        maior_ou_igual(
            f"fusao.nivel_{nivel}.score >= regras.nivel_{nivel}.score",
            float(niveis_fusao[nivel]["score"]),
            float(niveis_regras[nivel]["score"]),
        )
        menor_ou_igual(
            f"fusao.nivel_{nivel}.tau <= regras.nivel_{nivel}.tau",
            float(niveis_fusao[nivel]["tau"]),
            float(niveis_regras[nivel]["tau"]),
        )
    return checagens


def _deltas_numericos(atual: Any, referencia: Any) -> Any:
    if isinstance(atual, Mapping) and isinstance(referencia, Mapping):
        return {
            str(chave): delta
            for chave in sorted(set(atual) & set(referencia), key=str)
            if (delta := _deltas_numericos(atual[chave], referencia[chave])) is not None
        }
    if (
        isinstance(atual, Real)
        and not isinstance(atual, bool)
        and isinstance(referencia, Real)
        and not isinstance(referencia, bool)
    ):
        return float(atual) - float(referencia)
    return None


def _contar_origens(candidatos: list[dict]) -> dict[str, int]:
    contagem = Counter(
        candidato.get("evidencia_deteccao", {}).get("origem", "sem_evidencia")
        for candidato in candidatos
    )
    return {origem: contagem[origem] for origem in sorted(contagem)}


def _resolver_documento(
    documento_id: str,
    texto: str,
    candidatos: list[dict],
    resolvedor: Any,
    *,
    eventos: list[dict] | None = None,
    decodificador: str | None = None,
) -> dict:
    try:
        return p.resolver_candidatos(
            documento_id,
            texto,
            candidatos,
            resolvedor,
            eventos_deteccao=eventos,
            decodificador_bio=decodificador,
            erros_processamento=None,
        )
    except Exception as erro:
        raise ErroGate(
            f"falha fatal ao finalizar candidatos de {documento_id}"
        ) from erro


def executar_gate(
    *,
    pasta_txt: Path,
    goldenset: Path,
    db: Path,
    modelo: Path,
    manifesto_avaliacao: Path,
    manifesto_treino: Path,
    saida: Path,
    limiar_lei: float,
    limiar_jur: float,
    detector_factory: Callable[..., Any] | None = None,
    scorer: Callable[[Path, Path], Mapping[str, Any]] | None = None,
    resolvedor: Any | None = None,
) -> dict[str, Any]:
    """Executa as três ablações e falha se a fusão regredir contra regras.

    ``detector_factory`` e ``scorer`` são pontos de injeção para testes. A
    fábrica sempre recebe ``permitir_download=False``.
    """
    limiares = {
        "LEI": _validar_limiar("limiar_lei", limiar_lei),
        "JUR": _validar_limiar("limiar_jur", limiar_jur),
    }

    # Esta auditoria deve preceder qualquer inicialização do modelo/GPU.
    avaliacao = carregar_manifesto(manifesto_avaliacao)
    treino = carregar_manifesto(manifesto_treino)
    garantir_sem_vazamento(avaliacao)
    garantir_sem_vazamento(treino)
    auditoria = auditar_sobreposicao(avaliacao, treino)
    if not auditoria["aprovado"]:
        raise ErroGate(
            "dataset de avaliação se sobrepõe ao treino por ID ou SHA-256"
        )

    textos = _carregar_textos(pasta_txt, avaliacao)
    for caminho, descricao in ((goldenset, "goldenset"), (db, "banco"), (modelo, "modelo")):
        if not caminho.exists():
            raise ErroGate(f"{descricao} inexistente: {caminho}")
    if saida.exists():
        if not saida.is_dir():
            raise ErroGate(f"saída deve ser uma pasta: {saida}")
        if any(saida.iterdir()):
            raise ErroGate(f"pasta de saída não está vazia: {saida}")
    proveniencia = {
        "commit_git": _commit_git(),
        "working_tree_limpa": _worktree_limpa(),
        "manifesto_avaliacao_sha256": _sha256(manifesto_avaliacao),
        "manifesto_treino_sha256": _sha256(manifesto_treino),
        "goldenset_sha256": _sha256(goldenset),
        "banco_sha256": _sha256(db),
        "modelo_sha256": _sha256_artefato(modelo),
        "config_pipeline_sha256": _sha256(RAIZ / "config" / "pipeline.yaml"),
    }

    if detector_factory is None:
        from ml_citacoes.inferencia import DetectorBIO

        detector_factory = DetectorBIO.de_artefato
    try:
        detector = detector_factory(modelo, permitir_download=False)
    except Exception as erro:
        raise ErroGate("não foi possível carregar o detector local") from erro
    decodificador = getattr(detector, "decodificador", None)
    if not isinstance(decodificador, str) or not decodificador:
        raise ErroGate("detector não informa o decodificador BIO")

    conexao = None
    if resolvedor is None:
        conexao = sqlite3.connect(u._uri_sqlite(db.resolve()) + "?immutable=1", uri=True)
        resolvedor = Resolvedor(conexao)

    documentos: dict[str, list[dict]] = {modo: [] for modo in MODOS}
    spans_modelo = 0
    chamadas_detector = 0
    origens: dict[str, Counter] = {"modelo": Counter(), "fusao": Counter()}
    try:
        for documento_id, texto in textos.items():
            regras = filtrar_distratores(extrair(texto), texto)
            try:
                predicoes = list(detector.detectar(texto))
            except Exception as erro:
                raise ErroGate(
                    f"inferência falhou em {documento_id}; fallback para regras é proibido"
                ) from erro
            chamadas_detector += 1
            spans_modelo += len(predicoes)

            resultado_modelo = fusionar(
                [], predicoes, texto, limiares=limiares, filtro_distrator=e_distrator
            )
            resultado_fusao = fusionar(
                regras, predicoes, texto, limiares=limiares, filtro_distrator=e_distrator
            )
            candidatos_por_modo = {
                "regras": [dict(item) for item in regras],
                "modelo": list(resultado_modelo.candidatos),
                "fusao": list(resultado_fusao.candidatos),
            }
            eventos_por_modo = {
                "regras": None,
                "modelo": list(resultado_modelo.eventos),
                "fusao": list(resultado_fusao.eventos),
            }
            for modo in ("modelo", "fusao"):
                origens[modo].update(_contar_origens(candidatos_por_modo[modo]))

            for modo in MODOS:
                documento = _resolver_documento(
                    documento_id,
                    texto,
                    candidatos_por_modo[modo],
                    resolvedor,
                    eventos=eventos_por_modo[modo],
                    decodificador=None if modo == "regras" else decodificador,
                )
                if modo != "regras":
                    if documento.get("decodificador_bio") != decodificador:
                        raise ErroGate(
                            f"{modo} sem confirmação de execução do detector em {documento_id}"
                        )
                    if "eventos_deteccao" not in documento:
                        raise ErroGate(
                            f"{modo} sem eventos do detector em {documento_id}; possível fallback"
                        )
                documentos[modo].append(documento)
    finally:
        if conexao is not None:
            conexao.close()

    if chamadas_detector != len(textos):
        raise ErroGate("detector não foi executado exatamente uma vez por documento")
    if spans_modelo == 0:
        raise ErroGate("detector não produziu nenhum span no conjunto; possível colapso ou no-op")

    if scorer is None:
        from steps.avaliar_metrica import avaliar as scorer

    saida.mkdir(parents=True, exist_ok=True)
    metricas: dict[str, Mapping[str, Any]] = {}
    resumo_modos: dict[str, dict[str, Any]] = {}
    for modo in MODOS:
        csv = gravar_saida(documentos[modo], saida / modo)
        try:
            resultado = scorer(csv, goldenset)
        except Exception as erro:
            raise ErroGate(f"scorer rejeitou a submissão do modo {modo}") from erro
        _metricas_niveis(resultado)
        metricas[modo] = resultado
        resumo_modos[modo] = {
            "citacoes": sum(len(documento["citacoes"]) for documento in documentos[modo]),
            "submission_sha256": _sha256(csv),
            "metricas": resultado,
        }

    checagens = _checagens_nao_regressao(metricas["regras"], metricas["fusao"])
    aprovado = all(item["aprovado"] for item in checagens)
    relatorio = {
        "versao": VERSAO_RELATORIO,
        "aprovado": aprovado,
        "parametros": {
            "limiares": limiares,
            "decodificador_bio": decodificador,
            "offline": True,
        },
        "proveniencia": proveniencia,
        "auditoria_sobreposicao": auditoria,
        "execucao": {
            "documentos": len(textos),
            "chamadas_detector": chamadas_detector,
            "spans_modelo_brutos": spans_modelo,
            "fallbacks": 0,
            "origens_candidatos": {
                modo: {origem: origens[modo][origem] for origem in sorted(origens[modo])}
                for modo in ("modelo", "fusao")
            },
        },
        "modos": resumo_modos,
        "delta_fusao_menos_regras": _deltas_numericos(
            metricas["fusao"], metricas["regras"]
        ),
        "checagens": checagens,
    }
    relatorio = _normalizar_json(relatorio)
    _gravar_json_deterministico(relatorio, saida / "relatorio.json")
    if not aprovado:
        falhas = [item["criterio"] for item in checagens if not item["aprovado"]]
        raise ErroGate(
            "fusão regrediu contra regras: " + "; ".join(falhas),
            relatorio=relatorio,
        )
    return relatorio


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt", type=Path, required=True)
    parser.add_argument("--goldenset", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--modelo", type=Path, required=True)
    parser.add_argument("--manifesto-avaliacao", type=Path, required=True)
    parser.add_argument("--manifesto-treino", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--limiar-lei", type=float, required=True)
    parser.add_argument("--limiar-jur", type=float, required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        relatorio = executar_gate(
            pasta_txt=args.txt,
            goldenset=args.goldenset,
            db=args.db,
            modelo=args.modelo,
            manifesto_avaliacao=args.manifesto_avaliacao,
            manifesto_treino=args.manifesto_treino,
            saida=args.saida,
            limiar_lei=args.limiar_lei,
            limiar_jur=args.limiar_jur,
        )
    except (ErroGate, ErroManifesto) as erro:
        parser.exit(1, f"erro: {erro}\n")
    print(
        f"gate híbrido aprovado: score={relatorio['modos']['fusao']['metricas']['score_final']:.6f}"
    )


if __name__ == "__main__":
    main()
