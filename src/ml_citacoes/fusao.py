"""Fusão conservadora: regras são preservadas e o modelo apenas acrescenta."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from .contratos import CitacaoModelo, EvidenciaDeteccao, ResultadoVerificacao


class VerificadorCandidatos(Protocol):
    def classificar_muitos(self, trechos: Sequence[str]) -> list[ResultadoVerificacao]: ...


@dataclass(frozen=True, slots=True)
class ResultadoFusao:
    candidatos: tuple[dict, ...]
    eventos: tuple[dict, ...]


_SEPARADOR_FRAGMENTOS = re.compile(r"^[\s,:\-\u2013\u2014]{0,8}$")
_TAMANHO_MAXIMO_COMBINADO = 220


def iou(a_inicio: int, a_fim: int, b_inicio: int, b_fim: int) -> float:
    intersecao = max(0, min(a_fim, b_fim) - max(a_inicio, b_inicio))
    if not intersecao:
        return 0.0
    return intersecao / (max(a_fim, b_fim) - min(a_inicio, b_inicio))


def cobertura_da_predicao(
    regra_inicio: int, regra_fim: int, pred_inicio: int, pred_fim: int
) -> float:
    """Fração do span do modelo contida no candidato produzido por regra."""
    largura_predicao = pred_fim - pred_inicio
    if largura_predicao <= 0:
        return 0.0
    intersecao = max(
        0,
        min(regra_fim, pred_fim) - max(regra_inicio, pred_inicio),
    )
    return intersecao / largura_predicao


def fusionar(
    regras: list[dict],
    modelo: Iterable[CitacaoModelo],
    texto: str,
    *,
    limiares: dict[str, float],
    filtro_distrator: Callable[[dict, str], bool],
    verificador: VerificadorCandidatos | None = None,
    iou_minimo: float = 0.5,
    cobertura_predicao_minima: float = 0.9,
    iou_nms: float = 0.5,
) -> ResultadoFusao:
    candidatos = [dict(item) for item in regras]
    eventos: list[dict] = []
    somente_modelo: list[tuple[CitacaoModelo, dict]] = []
    usados_por_regra: set[int] = set()
    predicoes = list(modelo)

    for indice_regra, regra in enumerate(regras):
        relacoes = [
            (
                indice,
                pred,
                iou(regra["inicio"], regra["fim"], pred.inicio, pred.fim),
                cobertura_da_predicao(
                    regra["inicio"], regra["fim"], pred.inicio, pred.fim
                ),
            )
            for indice, pred in enumerate(predicoes)
        ]
        sobrepostos = [
            item
            for item in relacoes
            if item[2] >= iou_minimo
            or (
                item[1].tipo_pipeline == regra["tipo"]
                and item[3] >= cobertura_predicao_minima
            )
        ]
        usados_por_regra.update(item[0] for item in sobrepostos)
        mesmo_tipo = [item for item in sobrepostos if item[1].tipo_pipeline == regra["tipo"]]
        if mesmo_tipo:
            indice, pred, valor_iou, _ = max(mesmo_tipo, key=lambda item: item[2])
            evidencia = EvidenciaDeteccao(
                "ambos", pred.tipo_pipeline, pred.score_bruto, pred.score_calibrado,
                valor_iou, "concordancia", pred.transformacoes_ocr,
            )
        elif sobrepostos:
            indice, pred, valor_iou, _ = max(sobrepostos, key=lambda item: item[2])
            evidencia = EvidenciaDeteccao(
                "regra", pred.tipo_pipeline, pred.score_bruto, pred.score_calibrado,
                valor_iou, "discordancia_tipo", pred.transformacoes_ocr,
            )
        else:
            evidencia = EvidenciaDeteccao("regra", relacao="somente_regra")
        candidatos[indice_regra]["evidencia_deteccao"] = evidencia.como_dict()
        eventos.append(evidencia.como_dict() | {"inicio": regra["inicio"], "fim": regra["fim"]})

    for indice, pred in enumerate(predicoes):
        if indice in usados_por_regra:
            continue
        score = pred.score_calibrado if pred.score_calibrado is not None else pred.score_bruto
        if score < limiares[pred.tipo]:
            continue
        candidato = {
            "tipo": pred.tipo_pipeline,
            "fonte": "modelo",
            "inicio": pred.inicio,
            "fim": pred.fim,
            "trecho": texto[pred.inicio:pred.fim],
            "pontuacao": 0,
            "registros": [],
        }
        # O verificador substitui decisões semânticas e limiares conservadores,
        # mas não os filtros estruturais de OAB, folhas, valores e cabeçalho.
        if filtro_distrator(candidato, texto):
            continue
        evidencia = EvidenciaDeteccao(
            "modelo", pred.tipo_pipeline, pred.score_bruto, pred.score_calibrado,
            None, "somente_modelo", pred.transformacoes_ocr,
        )
        candidato["evidencia_deteccao"] = evidencia.como_dict()
        somente_modelo.append((pred, candidato))

    if verificador is not None and somente_modelo:
        somente_modelo.extend(
            _compor_fragmentos_vizinhos(somente_modelo, texto, filtro_distrator)
        )
        verificacoes = verificador.classificar_muitos(
            [candidato["trecho"] for _, candidato in somente_modelo]
        )
        if len(verificacoes) != len(somente_modelo):
            raise RuntimeError("verificador retornou quantidade inesperada de decisões")
        avaliados: list[tuple[CitacaoModelo, dict, ResultadoVerificacao]] = []
        for (pred, candidato), verificacao in zip(somente_modelo, verificacoes):
            candidato["evidencia_deteccao"]["verificador"] = verificacao.como_dict()
            tipo_verificado = verificacao.tipo_pipeline
            if tipo_verificado is not None:
                candidato["tipo"] = tipo_verificado
            eventos.append(candidato["evidencia_deteccao"] | {
                "inicio": pred.inicio,
                "fim": pred.fim,
                "aceito": verificacao.aceita,
                "tipo_final": candidato["tipo"] if verificacao.aceita else None,
            })
            avaliados.append((pred, candidato, verificacao))
        somente_modelo = _selecionar_verificados(avaliados)

    for pred, candidato in _nms(somente_modelo, iou_nms):
        candidatos.append(candidato)
        if verificador is None:
            eventos.append(candidato["evidencia_deteccao"] | {"inicio": pred.inicio, "fim": pred.fim})
    candidatos.sort(key=lambda item: (item["inicio"], item["fim"], item["tipo"]))
    return ResultadoFusao(tuple(candidatos), tuple(eventos))


def _nms(itens: list[tuple[CitacaoModelo, dict]], limite: float) -> list[tuple[CitacaoModelo, dict]]:
    ordenados = sorted(
        itens,
        key=lambda item: item[0].score_calibrado if item[0].score_calibrado is not None else item[0].score_bruto,
        reverse=True,
    )
    mantidos: list[tuple[CitacaoModelo, dict]] = []
    for item in ordenados:
        pred = item[0]
        if any(
            pred.tipo == outro[0].tipo and iou(pred.inicio, pred.fim, outro[0].inicio, outro[0].fim) >= limite
            for outro in mantidos
        ):
            continue
        mantidos.append(item)
    return mantidos


def _score_modelo(pred: CitacaoModelo) -> float:
    return pred.score_calibrado if pred.score_calibrado is not None else pred.score_bruto


def _compor_fragmentos_vizinhos(
    itens: list[tuple[CitacaoModelo, dict]],
    texto: str,
    filtro_distrator: Callable[[dict, str], bool],
) -> list[tuple[CitacaoModelo, dict]]:
    """Cria hipóteses adjacentes para a LLM, sem alterar os spans originais.

    A composição só existe no caminho com verificador. Conectores com letras ou
    ponto e vírgula não são aceitos, evitando atravessar ``e o Decr.`` ou o
    separador mais comum entre duas citações independentes.
    """
    ordenados = sorted(itens, key=lambda item: (item[0].inicio, item[0].fim))
    combinados: list[tuple[CitacaoModelo, dict]] = []
    vistos: set[tuple[int, int]] = set()

    for (esquerdo, _), (direito, _) in zip(ordenados, ordenados[1:]):
        if direito.inicio < esquerdo.fim:
            continue
        intervalo = texto[esquerdo.fim:direito.inicio]
        if not _SEPARADOR_FRAGMENTOS.fullmatch(intervalo):
            continue
        if direito.fim - esquerdo.inicio > _TAMANHO_MAXIMO_COMBINADO:
            continue
        limites = (esquerdo.inicio, direito.fim)
        if limites in vistos:
            continue
        vistos.add(limites)

        predominante = max((esquerdo, direito), key=_score_modelo)
        calibrado = (
            min(esquerdo.score_calibrado, direito.score_calibrado)
            if esquerdo.score_calibrado is not None and direito.score_calibrado is not None
            else None
        )
        combinado = CitacaoModelo(
            esquerdo.inicio,
            direito.fim,
            predominante.tipo,
            min(esquerdo.score_bruto, direito.score_bruto),
            calibrado,
        )
        origens = [(esquerdo.inicio, esquerdo.fim), (direito.inicio, direito.fim)]
        evidencia = EvidenciaDeteccao(
            "modelo",
            combinado.tipo_pipeline,
            combinado.score_bruto,
            combinado.score_calibrado,
            None,
            "somente_modelo",
        ).como_dict()
        evidencia["composicao_fragmentos"] = [
            {"inicio": inicio, "fim": fim} for inicio, fim in origens
        ]
        candidato = {
            "tipo": combinado.tipo_pipeline,
            "fonte": "modelo",
            "inicio": combinado.inicio,
            "fim": combinado.fim,
            "trecho": texto[combinado.inicio:combinado.fim],
            "pontuacao": 0,
            "registros": [],
            "evidencia_deteccao": evidencia,
            "_fragmentos_origem": origens,
        }
        if not filtro_distrator(candidato, texto):
            combinados.append((combinado, candidato))
    return combinados


def _selecionar_verificados(
    avaliados: list[tuple[CitacaoModelo, dict, ResultadoVerificacao]],
) -> list[tuple[CitacaoModelo, dict]]:
    """Aceita combinação apenas se todos os fragmentos isolados foram rejeitados."""
    decisoes_individuais = {
        (pred.inicio, pred.fim): verificacao
        for pred, candidato, verificacao in avaliados
        if "_fragmentos_origem" not in candidato
    }
    aceitos: list[tuple[CitacaoModelo, dict]] = []
    for pred, candidato, verificacao in avaliados:
        if not verificacao.aceita:
            continue
        origens = candidato.get("_fragmentos_origem")
        if origens and any(
            decisoes_individuais.get(tuple(origem)) is not None
            and decisoes_individuais[tuple(origem)].aceita
            for origem in origens
        ):
            continue
        candidato.pop("_fragmentos_origem", None)
        aceitos.append((pred, candidato))
    return aceitos
