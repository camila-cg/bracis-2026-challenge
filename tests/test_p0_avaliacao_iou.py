"""Regressões do alinhamento interno com a métrica oficial."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from detectar_citacoes import avaliar_documento


def _candidato(fim: int, inicio: int = 0, tipo: str = "lei") -> dict:
    return {
        "inicio": inicio,
        "fim": fim,
        "tipo": tipo,
        "trecho": "x" * (fim - inicio),
    }


def _anotacao() -> dict:
    return {
        "citacao_id": "g1",
        "inicio": "0",
        "fim": "100",
        "tipo": "lei",
    }


def test_iou_abaixo_de_meio_nao_casa():
    resultado = avaliar_documento([_candidato(49)], [_anotacao()])

    assert resultado["verdadeiros_positivos"] == 0
    assert resultado["falsos_positivos"] == 1
    assert resultado["falsos_negativos"] == 1


def test_iou_igual_a_meio_casa():
    resultado = avaliar_documento([_candidato(50)], [_anotacao()])

    assert resultado["verdadeiros_positivos"] == 1
    assert resultado["falsos_positivos"] == 0
    assert resultado["falsos_negativos"] == 0


def test_matching_prefere_maior_iou_sem_reutilizar_predicao():
    candidatos = [_candidato(80), _candidato(130, inicio=30)]
    resultado = avaliar_documento(candidatos, [_anotacao()])

    assert resultado["verdadeiros_positivos"] == 1
    assert resultado["falsos_positivos"] == 1
    assert resultado["indices_correspondentes"] == {0}


def test_componente_extra_quase_contido_em_gold_casado_e_ignorado():
    candidatos = [_candidato(100), _candidato(20, inicio=10)]
    resultado = avaliar_documento(candidatos, [_anotacao()])

    assert resultado["verdadeiros_positivos"] == 1
    assert resultado["falsos_positivos"] == 0
    assert resultado["extras_ignorados"] == 1
    assert resultado["indices_extras_ignorados"] == {1}


def test_matching_oficial_independe_do_tipo_interno():
    resultado = avaliar_documento(
        [_candidato(100, tipo="jurisprudencia")],
        [_anotacao()],
    )

    assert resultado["verdadeiros_positivos"] == 1
    assert resultado["falsos_positivos"] == 0


def test_duplicata_invalida_nao_e_perdoada_como_extra():
    resultado = avaliar_documento(
        [_candidato(100), _candidato(100)],
        [_anotacao()],
    )

    assert resultado["verdadeiros_positivos"] == 1
    assert resultado["falsos_positivos"] == 1
    assert resultado["extras_ignorados"] == 0
    assert resultado["duplicatas_invalidas"] == 1
    assert resultado["submissao_valida"] is False
