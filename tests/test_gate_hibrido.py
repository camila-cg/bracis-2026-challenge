"""Contrato do gate pareado, sem carregar pesos ou acessar a rede."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_citacoes.contratos import CitacaoModelo
from validacao import gate_hibrido as g
from validacao import manifesto_datasets as m


def _manifesto(
    raiz: Path, nome: str, papel: str, documentos: dict[str, str]
) -> tuple[Path, Path]:
    pasta = raiz / nome / "txt"
    pasta.mkdir(parents=True)
    for documento_id, texto in documentos.items():
        (pasta / f"{documento_id}.txt").write_text(texto, encoding="utf-8", newline="\n")
    manifesto = m.gerar_manifesto(
        raiz,
        [
            {
                "nome": nome,
                "documentos": pasta.relative_to(raiz).as_posix(),
                "anotacoes": None,
                "origem": f"teste-{nome}",
                "papel": papel,
            }
        ],
    )
    caminho = raiz / nome / "manifesto.json"
    m.gravar_manifesto(manifesto, caminho)
    return caminho, pasta


def _metricas(score: float, *, tau: float = 0.0) -> dict:
    nivel = {
        "score": score,
        "macro_f1": min(score, 1.0),
        "f1_por_classe": {
            "real": min(score, 1.0),
            "inventada": min(score, 1.0),
            "incompleta": min(score, 1.0),
        },
        "tau": tau,
        "s": min(score, 1.0),
        "b": 0.05,
    }
    return {"score_final": score, "niveis": {1: nivel, 2: dict(nivel)}}


class DetectorFalso:
    decodificador = "viterbi"

    def __init__(self, *, falhar: bool = False, vazio: bool = False):
        self.chamadas = 0
        self.falhar = falhar
        self.vazio = vazio

    def detectar(self, texto: str):
        self.chamadas += 1
        if self.falhar:
            raise RuntimeError("GPU indisponível")
        if self.vazio:
            return []
        inicio = texto.index("Modelo")
        return [CitacaoModelo(inicio, inicio + len("Modelo"), "LEI", 0.99)]


class FabricaDetector:
    def __init__(self, detector: DetectorFalso):
        self.detector = detector
        self.chamadas = 0
        self.permitir_download = None

    def __call__(self, _modelo: Path, *, permitir_download: bool):
        self.chamadas += 1
        self.permitir_download = permitir_download
        return self.detector


class ScorerFalso:
    def __init__(self, scores: dict[str, dict] | None = None):
        self.scores = scores or {
            "regras": _metricas(0.80),
            "modelo": _metricas(0.60),
            "fusao": _metricas(0.85),
        }
        self.chamadas: list[str] = []

    def __call__(self, submission: Path, _goldenset: Path):
        modo = submission.parent.name
        self.chamadas.append(modo)
        return self.scores[modo]


def _finalizar_candidatos(
    documento_id,
    _texto,
    candidatos,
    _resolvedor,
    *,
    eventos_deteccao=None,
    decodificador_bio=None,
    erros_processamento=None,
):
    assert erros_processamento is None
    citacoes = []
    for candidato in candidatos:
        citacao = {
            "inicio": candidato["inicio"],
            "fim": candidato["fim"],
            "trecho": candidato["trecho"],
            "tipo": candidato["tipo"],
            "classificacao": "incompleta",
            "resolucao": None,
            "confianca": 0.8,
        }
        if "evidencia_deteccao" in candidato:
            citacao["evidencia_deteccao"] = candidato["evidencia_deteccao"]
        citacoes.append(citacao)
    documento = {"documento_id": documento_id, "citacoes": citacoes}
    if eventos_deteccao is not None:
        documento["eventos_deteccao"] = eventos_deteccao
    if decodificador_bio is not None:
        documento["decodificador_bio"] = decodificador_bio
    return documento


@pytest.fixture
def ambiente(tmp_path: Path, monkeypatch):
    avaliacao, textos = _manifesto(
        tmp_path,
        "avaliacao",
        "teste",
        {
            "caso_n1_001": "Regra -- Modelo",
            "caso_n2_002": "Regra ++ Modelo",
        },
    )
    treino, _ = _manifesto(
        tmp_path, "treino", "treino", {"treino_n1_001": "conteúdo diferente"}
    )
    goldenset = tmp_path / "goldenset.csv"
    goldenset.write_text("documento_id,inicio,fim,trecho,tipo,classificacao,id_canonico\n")
    db = tmp_path / "base.db"
    db.touch()
    modelo = tmp_path / "modelo"
    modelo.mkdir()

    def extrair_regra(texto: str):
        return [{"inicio": 0, "fim": 5, "trecho": texto[:5], "tipo": "lei"}]

    monkeypatch.setattr(g, "extrair", extrair_regra)
    monkeypatch.setattr(g, "filtrar_distratores", lambda itens, _texto: list(itens))
    monkeypatch.setattr(g, "e_distrator", lambda _item, _texto: False)
    monkeypatch.setattr(g.p, "resolver_candidatos", _finalizar_candidatos)

    return SimpleNamespace(
        raiz=tmp_path,
        avaliacao=avaliacao,
        treino=treino,
        textos=textos,
        goldenset=goldenset,
        db=db,
        modelo=modelo,
    )


def _executar(
    ambiente,
    *,
    detector: DetectorFalso | None = None,
    scorer: ScorerFalso | None = None,
    saida: Path | None = None,
    limiar_lei: float = 0.95,
    limiar_jur: float = 0.95,
):
    detector = detector or DetectorFalso()
    fabrica = FabricaDetector(detector)
    scorer = scorer or ScorerFalso()
    relatorio = g.executar_gate(
        pasta_txt=ambiente.textos,
        goldenset=ambiente.goldenset,
        db=ambiente.db,
        modelo=ambiente.modelo,
        manifesto_avaliacao=ambiente.avaliacao,
        manifesto_treino=ambiente.treino,
        saida=saida or ambiente.raiz / "saida",
        limiar_lei=limiar_lei,
        limiar_jur=limiar_jur,
        detector_factory=fabrica,
        scorer=scorer,
        resolvedor=object(),
    )
    return relatorio, detector, fabrica, scorer


def test_executa_detector_uma_vez_por_documento_e_gera_tres_submissoes(ambiente):
    relatorio, detector, fabrica, scorer = _executar(ambiente)

    assert detector.chamadas == 2
    assert fabrica.chamadas == 1
    assert fabrica.permitir_download is False
    assert scorer.chamadas == ["regras", "modelo", "fusao"]
    assert relatorio["execucao"]["chamadas_detector"] == 2
    assert relatorio["execucao"]["fallbacks"] == 0
    assert relatorio["aprovado"] is True
    assert len(relatorio["proveniencia"]["banco_sha256"]) == 64
    assert len(relatorio["proveniencia"]["modelo_sha256"]) == 64
    assert len(relatorio["proveniencia"]["config_pipeline_sha256"]) == 64
    for modo in g.MODOS:
        assert (ambiente.raiz / "saida" / modo / "submission.csv").is_file()


def test_relatorio_e_deterministico_em_diretorios_diferentes(ambiente):
    primeiro, *_ = _executar(ambiente, saida=ambiente.raiz / "saida-a")
    segundo, *_ = _executar(ambiente, saida=ambiente.raiz / "saida-b")

    assert primeiro == segundo
    assert (ambiente.raiz / "saida-a" / "relatorio.json").read_bytes() == (
        ambiente.raiz / "saida-b" / "relatorio.json"
    ).read_bytes()
    assert json.loads((ambiente.raiz / "saida-a" / "relatorio.json").read_text()) == primeiro


def test_vazamento_por_hash_aborta_antes_de_criar_detector(tmp_path: Path):
    avaliacao, textos = _manifesto(
        tmp_path, "avaliacao", "teste", {"id_renomeado": "mesmos bytes"}
    )
    treino, _ = _manifesto(
        tmp_path, "treino", "treino", {"id_original": "mesmos bytes"}
    )
    fabrica = FabricaDetector(DetectorFalso())

    with pytest.raises(g.ErroGate, match="sobrepõe ao treino"):
        g.executar_gate(
            pasta_txt=textos,
            goldenset=tmp_path / "ausente.csv",
            db=tmp_path / "ausente.db",
            modelo=tmp_path / "ausente-modelo",
            manifesto_avaliacao=avaliacao,
            manifesto_treino=treino,
            saida=tmp_path / "saida",
            limiar_lei=0.95,
            limiar_jur=0.95,
            detector_factory=fabrica,
            scorer=ScorerFalso(),
            resolvedor=object(),
        )

    assert fabrica.chamadas == 0


def test_excecao_do_detector_e_fatal_sem_fallback(ambiente):
    detector = DetectorFalso(falhar=True)

    with pytest.raises(g.ErroGate, match="fallback para regras é proibido"):
        _executar(ambiente, detector=detector)

    assert detector.chamadas == 1
    assert not (ambiente.raiz / "saida" / "regras" / "submission.csv").exists()


def test_detector_sem_spans_falha_como_colapso(ambiente):
    detector = DetectorFalso(vazio=True)

    with pytest.raises(g.ErroGate, match="nenhum span"):
        _executar(ambiente, detector=detector)

    assert detector.chamadas == 2


@pytest.mark.parametrize("valor", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_rejeita_limiar_invalido_antes_de_criar_detector(ambiente, valor):
    fabrica = FabricaDetector(DetectorFalso())

    with pytest.raises(g.ErroGate, match="limiar_lei"):
        g.executar_gate(
            pasta_txt=ambiente.textos,
            goldenset=ambiente.goldenset,
            db=ambiente.db,
            modelo=ambiente.modelo,
            manifesto_avaliacao=ambiente.avaliacao,
            manifesto_treino=ambiente.treino,
            saida=ambiente.raiz / "saida",
            limiar_lei=valor,
            limiar_jur=0.95,
            detector_factory=fabrica,
            scorer=ScorerFalso(),
            resolvedor=object(),
        )

    assert fabrica.chamadas == 0


@pytest.mark.parametrize(
    ("fusao", "mensagem"),
    [
        (_metricas(0.79), "score_final"),
        (
            {
                "score_final": 0.81,
                "niveis": {
                    1: _metricas(0.79)["niveis"][1],
                    2: _metricas(0.81)["niveis"][2],
                },
            },
            "nivel_1.score",
        ),
        (_metricas(0.81, tau=0.1), "tau"),
    ],
)
def test_gate_de_nao_regressao_falha_e_preserva_relatorio(ambiente, fusao, mensagem):
    scorer = ScorerFalso(
        {
            "regras": _metricas(0.80),
            "modelo": _metricas(0.60),
            "fusao": fusao,
        }
    )

    with pytest.raises(g.ErroGate, match=mensagem) as capturado:
        _executar(ambiente, scorer=scorer)

    assert capturado.value.relatorio["aprovado"] is False
    salvo = json.loads((ambiente.raiz / "saida" / "relatorio.json").read_text())
    assert salvo["aprovado"] is False


def test_scorer_precisa_retornar_componentes_completos(ambiente):
    incompleta = {
        "score_final": 0.8,
        "niveis": {1: {"score": 0.8}, 2: _metricas(0.8)["niveis"][2]},
    }
    scorer = ScorerFalso(
        {"regras": incompleta, "modelo": _metricas(0.6), "fusao": _metricas(0.8)}
    )

    with pytest.raises(g.ErroGate, match="métricas incompletas"):
        _executar(ambiente, scorer=scorer)


def test_scorer_precisa_cobrir_os_dois_niveis_do_desafio(ambiente):
    um_nivel = _metricas(0.8)
    um_nivel["niveis"].pop(2)
    scorer = ScorerFalso(
        {"regras": um_nivel, "modelo": _metricas(0.6), "fusao": _metricas(0.8)}
    )

    with pytest.raises(g.ErroGate, match="exatamente os níveis 1 e 2"):
        _executar(ambiente, scorer=scorer)
