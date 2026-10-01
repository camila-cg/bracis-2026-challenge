"""Fusão conservadora: regras são preservadas e o modelo apenas acrescenta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from .contratos import CitacaoModelo, EvidenciaDeteccao, ResultadoVerificacao


class VerificadorCandidatos(Protocol):
    def classificar_muitos(self, trechos: Sequence[str]) -> list[ResultadoVerificacao]: ...


@dataclass(frozen=True, slots=True)
class ResultadoFusao:
    candidatos: tuple[dict, ...]
    eventos: tuple[dict, ...]


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
        verificacoes = verificador.classificar_muitos(
            [candidato["trecho"] for _, candidato in somente_modelo]
        )
        if len(verificacoes) != len(somente_modelo):
            raise RuntimeError("verificador retornou quantidade inesperada de decisões")
        aceitos: list[tuple[CitacaoModelo, dict]] = []
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
            if verificacao.aceita:
                aceitos.append((pred, candidato))
        somente_modelo = aceitos

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
