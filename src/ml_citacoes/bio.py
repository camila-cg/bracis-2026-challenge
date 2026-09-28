"""Alinhamento BIO e reconstrução por offsets absolutos em codepoints."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Iterable, Sequence

from .contratos import ID2LABEL, LABEL2ID, CitacaoModelo

IGNORAR = -100


@dataclass(frozen=True, slots=True)
class DiagnosticoBIO:
    invalidas: int = 0
    fragmentos: int = 0


def spans_para_bio(offsets: Sequence[Sequence[int]], spans: Iterable[dict]) -> list[int]:
    """Converte spans absolutos anotados em IDs BIO; special/padding vira -100."""
    anotacoes = sorted(
        (int(s["inicio"]), int(s["fim"]), _tipo_bio(str(s["tipo"]))) for s in spans
    )
    for atual, seguinte in zip(anotacoes, anotacoes[1:]):
        if atual[1] > seguinte[0]:
            raise ValueError("anotações sobrepostas não podem ser representadas em BIO")
    saida: list[int] = []
    offsets_validos = [(int(o[0]), int(o[1])) for o in offsets if int(o[1]) > int(o[0])]
    inicio_janela = min((o[0] for o in offsets_validos), default=0)
    primeiro_por_span: set[int] = set()
    for bruto in offsets:
        inicio, fim = map(int, bruto)
        if fim <= inicio:
            saida.append(IGNORAR)
            continue
        rotulo = "O"
        for indice, (span_inicio, span_fim, tipo) in enumerate(anotacoes):
            if inicio < span_fim and fim > span_inicio:
                # Se a janela começa no meio do span, a primeira etiqueta deve
                # continuar como I. Em uma janela que contém o começo real,
                # o primeiro subtoken sobreposto recebe B.
                janela_contem_inicio = inicio_janela <= span_inicio
                prefixo = "B" if janela_contem_inicio and indice not in primeiro_por_span else "I"
                primeiro_por_span.add(indice)
                rotulo = f"{prefixo}-{tipo}"
                break
        saida.append(LABEL2ID[rotulo])
    return saida


def bio_para_spans(
    labels: Sequence[int | str],
    offsets: Sequence[Sequence[int]],
    scores: Sequence[float] | None = None,
) -> tuple[list[CitacaoModelo], DiagnosticoBIO]:
    """Reconstrói spans sem busca textual; I inválido inicia span e é contabilizado."""
    spans: list[CitacaoModelo] = []
    atual: dict | None = None
    invalidas = 0
    fragmentos = 0
    for indice, (rotulo_bruto, offset) in enumerate(zip(labels, offsets)):
        inicio, fim = map(int, offset)
        if fim <= inicio:
            continue
        rotulo = ID2LABEL[int(rotulo_bruto)] if isinstance(rotulo_bruto, int) else rotulo_bruto
        score = float(scores[indice]) if scores is not None else 1.0
        if rotulo == "O":
            if atual is not None:
                spans.append(_fechar(atual))
                atual = None
            continue
        prefixo, tipo = rotulo.split("-", 1)
        continua = atual is not None and atual["tipo"] == tipo and prefixo == "I"
        if prefixo == "I" and not continua:
            invalidas += 1
        if prefixo == "B" and atual is not None and atual["tipo"] == tipo:
            fragmentos += 1
        if not continua:
            if atual is not None:
                spans.append(_fechar(atual))
            atual = {"inicio": inicio, "fim": fim, "tipo": tipo, "scores": [score]}
        else:
            atual["fim"] = max(atual["fim"], fim)
            atual["scores"].append(score)
    if atual is not None:
        spans.append(_fechar(atual))
    return spans, DiagnosticoBIO(invalidas, fragmentos)


def combinar_logits_janelas(
    offsets_por_janela: Sequence[Sequence[Sequence[int]]],
    logits_por_janela: Sequence[Sequence[Sequence[float]]],
    *, decodificador: str = "argmax",
) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    """Média de logits por offset; decodifica uma vez no documento completo.

    Os scores são softmax local do rótulo escolhido, não marginais de Viterbi.
    """
    if decodificador not in {"argmax", "viterbi"}:
        raise ValueError("decodificador deve ser argmax ou viterbi")
    if len(offsets_por_janela) != len(logits_por_janela):
        raise ValueError("número de janelas divergente")
    acumulado: dict[tuple[int, int], list[list[float]]] = {}
    for offsets, logits in zip(offsets_por_janela, logits_por_janela):
        if len(offsets) != len(logits):
            raise ValueError("offsets e logits com comprimentos divergentes")
        for bruto, vetor in zip(offsets, logits):
            offset = tuple(map(int, bruto))
            if offset[1] <= offset[0]:
                continue
            if len(vetor) != len(LABEL2ID) or not all(isfinite(float(x)) for x in vetor):
                raise ValueError("logits devem conter cinco valores finitos")
            acumulado.setdefault(offset, []).append([float(x) for x in vetor])
    offsets_ordenados = sorted(acumulado)
    medias = []
    for offset in offsets_ordenados:
        vetores = acumulado[offset]
        media = [sum(v[i] for v in vetores) / len(vetores) for i in range(len(vetores[0]))]
        medias.append(media)
    labels = (decodificar_viterbi(medias) if decodificador == "viterbi"
              else [max(range(len(v)), key=v.__getitem__) for v in medias])
    scores = [_softmax(v)[label] for v, label in zip(medias, labels)]
    return offsets_ordenados, labels, scores


def decodificar_viterbi(logits: Sequence[Sequence[float]]) -> list[int]:
    """BIO estrito, transições válidas=0 e inválidas=-inf, sem parâmetros.

    I-X só sucede B-X/I-X. No início do documento só O/B são permitidos.
    Somar logits equivale a somar log-softmax para selecionar o caminho.
    Empates usam a ordem fixa dos IDs, com O primeiro.
    """
    if not len(logits):
        return []
    if any(len(v) != len(LABEL2ID) or not all(isfinite(float(x)) for x in v) for v in logits):
        raise ValueError("logits devem conter cinco valores finitos")
    anteriores = {0: range(5), 1: range(5), 2: (1, 2), 3: range(5), 4: (3, 4)}
    valores = [float(x) if i in (0, 1, 3) else float("-inf") for i, x in enumerate(logits[0])]
    caminhos = []
    for vetor in logits[1:]:
        pais = [max(anteriores[i], key=valores.__getitem__) for i in range(5)]
        valores = [valores[pais[i]] + float(vetor[i]) for i in range(5)]
        caminhos.append(pais)
    atual = max(range(5), key=valores.__getitem__)
    labels = [atual]
    for pais in reversed(caminhos):
        atual = pais[atual]
        labels.append(atual)
    return list(reversed(labels))


def _softmax(valores: Sequence[float]) -> list[float]:
    maior = max(valores)
    expoentes = [exp(valor - maior) for valor in valores]
    total = sum(expoentes)
    return [valor / total for valor in expoentes]


def _tipo_bio(tipo: str) -> str:
    normalizado = tipo.upper()
    if normalizado in {"LEI", "NORMATIVO", "LEGISLACAO", "LEGISLAÇÃO"}:
        return "LEI"
    if normalizado in {"JUR", "JURISPRUDENCIA", "JURISPRUDÊNCIA"}:
        return "JUR"
    raise ValueError(f"tipo de citação desconhecido: {tipo}")


def _fechar(atual: dict) -> CitacaoModelo:
    scores = atual["scores"]
    # O mínimo é conservador: todas as partes do span precisam estar seguras.
    score = min(scores)
    return CitacaoModelo(atual["inicio"], atual["fim"], atual["tipo"], score, None)
