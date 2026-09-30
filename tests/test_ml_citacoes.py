"""Testes do token classifier BIO sem baixar pesos nem exigir torch."""

import csv
import hashlib
import itertools
import random
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from ml_citacoes.aumento import aumentar_citacao_real
from ml_citacoes.bio import decodificar_viterbi, IGNORAR, bio_para_spans, combinar_logits_janelas, spans_para_bio
from ml_citacoes.contratos import CitacaoModelo, LABEL2ID, ResultadoVerificacao
from ml_citacoes.fusao import fusionar
from ml_citacoes.metricas import avaliar_spans
from ml_citacoes.selecionar_modelo import selecionar
from ml_citacoes.splits import criar_folds, preparar, validar_origem_original, validar_sem_vazamento
from ml_citacoes.treinar import (
    _amostrar_tokens_o,
    _argumentos_checkpointing,
    _equilibrar_janelas,
    _metricas_tokens,
    _pesos_classes,
)


class AlinhamentoBIO(unittest.TestCase):
    def test_unicode_acentos_quebra_de_linha_e_special_tokens(self):
        texto = "⚖️ Prévia\nart. 5º da Constituição"
        inicio = texto.index("art.")
        fim = len(texto)
        offsets = [(0, 0), (inicio, inicio + 4), (inicio + 5, inicio + 7), (inicio + 8, fim), (0, 0)]
        labels = spans_para_bio(offsets, [{"inicio": inicio, "fim": fim, "tipo": "lei"}])
        self.assertEqual(labels, [IGNORAR, LABEL2ID["B-LEI"], LABEL2ID["I-LEI"], LABEL2ID["I-LEI"], IGNORAR])
        spans, diagnostico = bio_para_spans(labels, offsets)
        self.assertEqual((spans[0].inicio, spans[0].fim, spans[0].tipo), (inicio, fim, "LEI"))
        self.assertEqual(diagnostico.invalidas, 0)

    def test_i_invalido_e_fragmentacao_sao_mensurados(self):
        spans, diagnostico = bio_para_spans(
            ["I-JUR", "B-JUR"], [(0, 4), (5, 10)], [0.9, 0.8]
        )
        self.assertEqual(len(spans), 2)
        self.assertEqual((diagnostico.invalidas, diagnostico.fragmentos), (1, 1))

    def test_span_na_fronteira_e_janelas_sobrepostas(self):
        offsets, labels, scores = combinar_logits_janelas(
            [[(0, 4), (5, 9), (10, 14)], [(5, 9), (10, 14), (15, 19)]],
            [
                [[0, 0, 0, 8, 0], [0, 0, 0, 0, 8], [0, 0, 0, 0, 8]],
                [[0, 0, 0, 0, 8], [0, 0, 0, 0, 8], [8, 0, 0, 0, 0]],
            ],
        )
        spans, _ = bio_para_spans(labels, offsets, scores)
        self.assertEqual([(s.inicio, s.fim, s.tipo) for s in spans], [(0, 14, "JUR")])

    def test_janela_que_comeca_no_meio_do_span_recebe_i(self):
        labels = spans_para_bio(
            [(10, 14), (15, 20)],
            [{"inicio": 5, "fim": 20, "tipo": "jurisprudencia"}],
        )
        self.assertEqual(labels, [LABEL2ID["I-JUR"], LABEL2ID["I-JUR"]])


class SplitsEAumento(unittest.TestCase):
    def test_fold_e_inteiramente_agrupado_por_documento(self):
        docs = [
            {"documento_id": f"doc_{i}", "texto": "x", "spans": [{"tipo": "lei" if i % 2 else "jurisprudencia", "classificacao": "real", "nivel": str(1 + i % 2)}]}
            for i in range(10)
        ]
        folds = criar_folds(docs, 5, 123)
        for fold in folds:
            validar_sem_vazamento(fold["treino"], fold["validacao"])
        self.assertEqual({x for f in folds for x in f["validacao"]}, {d["documento_id"] for d in docs})

    def test_vazamento_dispara_erro(self):
        with self.assertRaises(ValueError):
            validar_sem_vazamento(["a", "b"], ["b", "c"])

    def test_dataset_sintetico_e_recusado(self):
        with self.assertRaises(ValueError):
            validar_origem_original(Path("dataset_sintetico/txt"))

    def test_sinteticos_entram_so_no_treino_e_incluem_negativos(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            originais = raiz / "originais"
            originais.mkdir()
            gold_original = raiz / "gold_original.csv"
            campos = ["nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho", "tipo", "classificacao", "id_canonico"]
            with gold_original.open("w", encoding="utf-8", newline="") as arquivo:
                escritor = csv.DictWriter(arquivo, fieldnames=campos)
                escritor.writeheader()
                for indice in range(4):
                    documento_id = f"real_{indice}"
                    texto = "Texto art. 1 do Código Civil."
                    (originais / f"{documento_id}.txt").write_text(texto, encoding="utf-8")
                    inicio, fim = texto.index("art."), texto.index(".", texto.index("art."))
                    escritor.writerow({
                        "nivel": "1", "documento_id": documento_id, "citacao_id": f"c{indice}",
                        "inicio": inicio, "fim": fim, "trecho": texto[inicio:fim], "tipo": "lei",
                        "classificacao": "real", "id_canonico": "1",
                    })

            sintetico = raiz / "dataset_sintetico_corrigido"
            (sintetico / "txt").mkdir(parents=True)
            textos = {
                "jbsg_com_citacao": "Conforme Recurso Especial do STJ, de 2022, Rel. Min. X.",
                "jbsg_negativo": "Relatório administrativo sem qualquer citação jurídica.",
            }
            with (sintetico / "manifest.csv").open("w", encoding="utf-8", newline="") as arquivo:
                escritor = csv.DictWriter(arquivo, fieldnames=["nivel", "documento_id", "arquivo"])
                escritor.writeheader()
                for indice, (documento_id, texto) in enumerate(textos.items(), start=1):
                    caminho = f"txt/{documento_id}.txt"
                    (sintetico / caminho).write_text(texto, encoding="utf-8")
                    escritor.writerow({"nivel": str(indice), "documento_id": documento_id, "arquivo": caminho})
            texto_citado = textos["jbsg_com_citacao"]
            trecho = "Recurso Especial do STJ, de 2022, Rel. Min. X"
            inicio = texto_citado.index(trecho)
            with (sintetico / "goldenset_offsets.csv").open("w", encoding="utf-8", newline="") as arquivo:
                escritor = csv.DictWriter(arquivo, fieldnames=campos)
                escritor.writeheader()
                escritor.writerow({
                    "nivel": "1", "documento_id": "jbsg_com_citacao", "citacao_id": "s1",
                    "inicio": inicio, "fim": inicio + len(trecho), "trecho": trecho,
                    "tipo": "jurisprudencia", "classificacao": "incompleta", "id_canonico": "",
                })

            saida = raiz / "preparado"
            preparar(originais, gold_original, saida, n_folds=2, seed=7, sinteticos=[sintetico])
            manifesto = json.loads((saida / "folds.json").read_text(encoding="utf-8"))
            ids_sinteticos = set(textos)
            for fold in manifesto["folds"]:
                self.assertTrue(ids_sinteticos.issubset(fold["treino"]))
                self.assertTrue(ids_sinteticos.isdisjoint(fold["validacao"]))
            documentos = [json.loads(linha) for linha in (saida / "documentos.jsonl").read_text(encoding="utf-8").splitlines()]
            negativo = next(doc for doc in documentos if doc["documento_id"] == "jbsg_negativo")
            self.assertEqual(negativo["spans"], [])
            self.assertEqual(negativo["origem"], "sintetico")

    def test_ocr_nunca_troca_digito_por_digito(self):
        original = "REsp 1050"
        alterado, transformacoes = aumentar_citacao_real(original, seed=7, probabilidade=1.0)
        for transformacao in transformacoes:
            if transformacao.original.isdigit():
                self.assertFalse(transformacao.substituto.isdigit())
        self.assertNotEqual(original, alterado)

    def test_digito_isolado_nao_recebe_ruido_irrecuperavel(self):
        alterado, transformacoes = aumentar_citacao_real("art. 1º da CF", seed=1, probabilidade=1.0)
        self.assertEqual(alterado, "art. 1º da CF")
        self.assertEqual(transformacoes, ())


class FusaoConservadora(unittest.TestCase):
    @staticmethod
    def nao_distrator(candidato, texto):
        return False

    def test_concordancia_preserva_span_exato_da_regra(self):
        texto = "Antes REsp 123 do STJ depois"
        regra = {"tipo": "jurisprudencia", "inicio": 6, "fim": 22, "trecho": texto[6:22], "pontuacao": 7, "registros": ["original"]}
        modelo = CitacaoModelo(6, 25, "JUR", 0.99, 0.98)
        resultado = fusionar([regra], [modelo], texto, limiares={"LEI": 0.95, "JUR": 0.95}, filtro_distrator=self.nao_distrator)
        self.assertEqual((resultado.candidatos[0]["inicio"], resultado.candidatos[0]["fim"]), (6, 22))
        self.assertEqual(resultado.candidatos[0]["evidencia_deteccao"]["origem"], "ambos")

    def test_discordancia_de_tipo_nao_sobrescreve_regra(self):
        texto = "art. 5º da CF"
        regra = {"tipo": "lei", "inicio": 0, "fim": len(texto), "trecho": texto, "pontuacao": 8, "registros": []}
        resultado = fusionar([regra], [CitacaoModelo(0, len(texto), "JUR", 0.99)], texto, limiares={"LEI": 0.95, "JUR": 0.95}, filtro_distrator=self.nao_distrator)
        self.assertEqual(len(resultado.candidatos), 1)
        self.assertEqual(resultado.candidatos[0]["tipo"], "lei")
        self.assertEqual(resultado.candidatos[0]["evidencia_deteccao"]["relacao"], "discordancia_tipo")

    def test_modelo_so_acrescenta_acima_do_limiar_e_nms(self):
        texto = "jurisprudência pacífica desta Corte"
        modelo = [CitacaoModelo(0, len(texto), "JUR", 0.98), CitacaoModelo(0, len(texto) - 2, "JUR", 0.97)]
        resultado = fusionar([], modelo, texto, limiares={"LEI": 0.95, "JUR": 0.95}, filtro_distrator=self.nao_distrator)
        self.assertEqual(len(resultado.candidatos), 1)
        self.assertEqual(resultado.candidatos[0]["evidencia_deteccao"]["origem"], "modelo")

    def test_verificador_recebe_somente_exclusivos_e_incerto_nao_entra(self):
        class VerificadorFalso:
            def __init__(self):
                self.recebidos = None

            def classificar_muitos(self, trechos):
                self.recebidos = list(trechos)
                return [
                    ResultadoVerificacao("CITACAO_JURISPRUDENCIA", .8, {
                        "CITACAO_LEGISLACAO": .05, "CITACAO_JURISPRUDENCIA": .8,
                        "INCERTO": .1, "NAO_CITACAO": .05,
                    }, "falso", "hash"),
                    ResultadoVerificacao("INCERTO", .6, {
                        "CITACAO_LEGISLACAO": .1, "CITACAO_JURISPRUDENCIA": .1,
                        "INCERTO": .6, "NAO_CITACAO": .2,
                    }, "falso", "hash"),
                ]

        texto = "art. 5º da CF | Tema 1046 | dispositivo legal | OAB/BA 130394"
        regra = {"tipo": "lei", "inicio": 0, "fim": 13, "trecho": texto[:13]}
        inicio_tema = texto.index("Tema")
        inicio_vago = texto.index("dispositivo")
        inicio_oab = texto.index("OAB")
        modelo = [
            CitacaoModelo(0, 13, "LEI", .99),
            CitacaoModelo(inicio_tema, inicio_tema + len("Tema 1046"), "JUR", .2),
            CitacaoModelo(inicio_vago, inicio_vago + len("dispositivo legal"), "LEI", .1),
            CitacaoModelo(inicio_oab, len(texto), "JUR", .9),
        ]
        verificador = VerificadorFalso()
        resultado = fusionar(
            [regra], modelo, texto,
            limiares={"LEI": 0.0, "JUR": 0.0},
            filtro_distrator=lambda candidato, _: "OAB" in candidato["trecho"],
            verificador=verificador,
        )
        self.assertEqual(verificador.recebidos, ["Tema 1046", "dispositivo legal"])
        self.assertEqual([c["trecho"] for c in resultado.candidatos], [texto[:13], "Tema 1046"])
        eventos_modelo = [e for e in resultado.eventos if e.get("relacao") == "somente_modelo"]
        self.assertEqual([e["aceito"] for e in eventos_modelo], [True, False])
        self.assertEqual(eventos_modelo[1]["verificador"]["decisao"], "INCERTO")

    def test_verificador_aceito_corrige_tipo_do_lora(self):
        class VerificadorLegislacao:
            def classificar_muitos(self, trechos):
                return [ResultadoVerificacao("CITACAO_LEGISLACAO", .9, {
                    "CITACAO_LEGISLACAO": .9, "CITACAO_JURISPRUDENCIA": .05,
                    "INCERTO": .03, "NAO_CITACAO": .02,
                }, "falso", "hash")]

        texto = "Res. CNJ nº 455, de 27-4-2022"
        resultado = fusionar(
            [], [CitacaoModelo(0, len(texto), "JUR", .99)], texto,
            limiares={"LEI": 0.0, "JUR": 0.0},
            filtro_distrator=self.nao_distrator,
            verificador=VerificadorLegislacao(),
        )

        self.assertEqual(resultado.candidatos[0]["tipo"], "lei")
        self.assertEqual(resultado.eventos[0]["tipo_previsto"], "jurisprudencia")
        self.assertEqual(resultado.eventos[0]["tipo_final"], "lei")

    def test_classe_d_e_rejeitada(self):
        resultado = ResultadoVerificacao("NAO_CITACAO", .9, {
            "CITACAO_LEGISLACAO": .02, "CITACAO_JURISPRUDENCIA": .03,
            "INCERTO": .05, "NAO_CITACAO": .9,
        }, "falso", "hash")

        self.assertFalse(resultado.aceita)
        self.assertIsNone(resultado.tipo_pipeline)

    def test_verificador_com_quantidade_errada_falha_explicitamente(self):
        class Incompleto:
            def classificar_muitos(self, trechos):
                return []

        with self.assertRaisesRegex(RuntimeError, "quantidade inesperada"):
            fusionar(
                [], [CitacaoModelo(0, 4, "LEI", .1)], "texto",
                limiares={"LEI": 0.0, "JUR": 0.0},
                filtro_distrator=self.nao_distrator,
                verificador=Incompleto(),
            )


class VerificadorLLM(unittest.TestCase):
    def test_qwen_e_o_modelo_ollama_padrao(self):
        from ml_citacoes.verificador_llm import VerificadorOllama

        self.assertEqual(VerificadorOllama().modelo, "qwen3.6:latest")

    def test_logits_restritos_geram_decisao_e_confianca(self):
        from ml_citacoes.verificador_llm import PROMPT_SHA256, PROMPT_VERSAO, _resultado_de_logits

        resultado = _resultado_de_logits([3.0, 1.0, -2.0, -3.0], modelo="local")
        self.assertEqual(PROMPT_VERSAO, "zero_shot_v3_quatro_classes")
        self.assertEqual(resultado.decisao, "CITACAO_LEGISLACAO")
        self.assertEqual(resultado.modelo, "local")
        self.assertEqual(resultado.prompt_sha256, PROMPT_SHA256)
        self.assertAlmostEqual(sum(resultado.probabilidades.values()), 1.0)
        self.assertEqual(resultado.confianca, resultado.probabilidades["CITACAO_LEGISLACAO"])

    def test_empate_e_deterministico_e_logits_invalidos_sao_rejeitados(self):
        from ml_citacoes.verificador_llm import _resultado_de_logits, _softmax_restrito

        self.assertEqual(_resultado_de_logits([0, 0, 0, 0], modelo="x").decisao, "INCERTO")
        for valores in ([1, 2, 3], [1, float("nan"), 3, 4]):
            with self.assertRaises(ValueError):
                _softmax_restrito(valores)

    def test_resposta_ollama_usa_top_logprobs_das_quatro_classes(self):
        from ml_citacoes.verificador_llm import _resultado_ollama

        resposta = {"logprobs": [{
            "token": " A", "logprob": -0.1,
            "top_logprobs": [
                {"token": " A", "logprob": -0.1},
                {"token": " B", "logprob": -2.0},
                {"token": " C", "logprob": -3.0},
                {"token": " D", "logprob": -4.0},
            ],
        }]}
        resultado = _resultado_ollama(resposta, modelo="ollama:teste")
        self.assertEqual(resultado.decisao, "CITACAO_LEGISLACAO")
        self.assertEqual(resultado.modelo, "ollama:teste")
        with self.assertRaisesRegex(RuntimeError, "todos os rótulos"):
            _resultado_ollama({"logprobs": [{
                "token": " A", "logprob": -0.1, "top_logprobs": [],
            }]}, modelo="x")


class MetricasPorDocumento(unittest.TestCase):
    def test_offsets_iguais_em_documentos_diferentes_nao_casam(self):
        verdade = [{"documento_id": "a", "inicio": 10, "fim": 20, "tipo": "lei"}]
        previsto = [{"documento_id": "b", "inicio": 10, "fim": 20, "tipo": "LEI"}]
        geral = avaliar_spans(verdade, previsto)["geral"]
        self.assertEqual(geral, {
            "tp": 0, "fp": 1, "fn": 1,
            "precisao": 0.0, "recall": 0.0, "f1": 0.0,
        })

    def test_offsets_no_mesmo_documento_casam(self):
        verdade = [{"documento_id": "a", "inicio": 10, "fim": 20, "tipo": "lei"}]
        previsto = [{"documento_id": "a", "inicio": 10, "fim": 20, "tipo": "LEI"}]
        geral = avaliar_spans(verdade, previsto)["geral"]
        self.assertEqual((geral["tp"], geral["fp"], geral["fn"]), (1, 0, 0))


class Selecao(unittest.TestCase):
    def test_inconsistencia_mantem_bertimbau(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            legal, bert = [], []
            for i, (a, b) in enumerate(zip([.8, .9, .8, .9, .8], [.85] * 5)):
                for nome, valor, lista in (("l", a, legal), ("b", b, bert)):
                    pasta = raiz / f"{nome}{i}"
                    pasta.mkdir()
                    (pasta / "resultados_validacao.json").write_text(json.dumps({
                        "modelo": {"geral": {"f1": valor}},
                        "fusao": {"geral": {"f1": 1.0}},
                    }))
                    (pasta / "manifesto.json").write_text(json.dumps({
                        "fold": str(i), "seed": 1, "documentos_validacao": [f"doc{i}"], "dataset_sha256": "abc"
                    }))
                    lista.append(pasta)
            self.assertEqual(selecionar(legal, bert)["modelo_escolhido"], "bertimbau")

    def test_rejeita_dois_modelos_colapsados_mesmo_com_fusao_perfeita(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            modelos = {"legal": [], "bertimbau": []}
            for nome, pastas in modelos.items():
                for i in range(5):
                    pasta = raiz / f"{nome}{i}"
                    pasta.mkdir()
                    (pasta / "resultados_validacao.json").write_text(json.dumps({
                        "modelo": {"geral": {"f1": 0.0}},
                        "fusao": {"geral": {"f1": 1.0}},
                    }))
                    (pasta / "manifesto.json").write_text(json.dumps({
                        "fold": str(i), "seed": 1, "documentos_validacao": [f"doc{i}"], "dataset_sha256": "abc"
                    }))
                    pastas.append(pasta)
            with self.assertRaisesRegex(ValueError, "colapsaram"):
                selecionar(modelos["legal"], modelos["bertimbau"])

    def test_nunca_escolhe_modelo_colapsado_no_desempate(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            legal, bert = [], []
            for i in range(5):
                for nome, valor, lista in (("legal", 0.2, legal), ("bert", 0.0, bert)):
                    pasta = raiz / f"{nome}{i}"
                    pasta.mkdir()
                    (pasta / "resultados_validacao.json").write_text(json.dumps({
                        "modelo": {"geral": {"f1": valor}},
                        "fusao": {"geral": {"f1": 1.0}},
                    }))
                    (pasta / "manifesto.json").write_text(json.dumps({
                        "fold": str(i), "seed": 1, "documentos_validacao": [f"doc{i}"], "dataset_sha256": "abc"
                    }))
                    lista.append(pasta)
            self.assertEqual(selecionar(legal, bert)["modelo_escolhido"], "legal-bert")


class BalanceamentoTreino(unittest.TestCase):
    def test_gradient_checkpointing_e_explicito_e_nao_reentrante(self):
        self.assertEqual(
            _argumentos_checkpointing({"gradient_checkpointing": True}),
            {
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
            },
        )
        self.assertEqual(
            _argumentos_checkpointing({}),
            {"gradient_checkpointing": False},
        )

    def test_limita_janelas_negativas_e_preserva_positivas(self):
        positivas = [{"labels": [-100, 0, 1, 2]} for _ in range(2)]
        negativas = [{"labels": [-100, 0, 0]} for _ in range(10)]
        equilibradas, diagnostico = _equilibrar_janelas(positivas + negativas, 1.0, 7)
        self.assertEqual(len(equilibradas), 4)
        self.assertEqual(diagnostico["janelas_negativas_usadas"], 2)
        self.assertEqual(sum(any(x > 0 for x in item["labels"]) for item in equilibradas), 2)

    def test_pesos_dao_mais_importancia_a_rotulos_raros(self):
        exemplos = [{"labels": [0] * 100 + [1] * 2 + [2] * 8 + [3] * 2 + [4] * 8}]
        pesos, contagens = _pesos_classes(exemplos, potencia=0.5, peso_maximo=10.0)
        self.assertEqual(pesos[0], 1.0)
        self.assertGreater(pesos[1], pesos[2])
        self.assertGreater(pesos[3], pesos[4])
        self.assertEqual(contagens["B-LEI"], 2)

    def test_metrica_positiva_exige_rotulo_bio_correto(self):
        logits = np.zeros((1, 4, 5))
        logits[0, 0, 0] = 1
        logits[0, 1, 2] = 1  # prevê I-LEI onde o correto é B-LEI
        logits[0, 2, 2] = 1  # I-LEI correto
        logits[0, 3, 3] = 1  # falso positivo JUR
        labels = np.array([[0, 1, 2, 0]])
        metricas = _metricas_tokens((logits, labels))
        self.assertAlmostEqual(metricas["positive_precision"], 1 / 3)
        self.assertEqual(metricas["positive_recall"], 0.5)
        self.assertAlmostEqual(metricas["positive_f1"], 0.4)
        self.assertAlmostEqual(metricas["positive_f2"], 5 / 11)
        self.assertEqual(metricas["detection_recall"], 1.0)

    def test_amostragem_ignora_o_excedente_sem_remover_contexto(self):
        exemplos = [
            {"input_ids": list(range(10)), "labels": [0] * 8 + [1, 2]},
            {"input_ids": list(range(5)), "labels": [0] * 5},
        ]
        amostrados, diagnostico = _amostrar_tokens_o(exemplos, proporcao_o=2.0, seed=7)
        self.assertEqual(amostrados[0]["input_ids"], exemplos[0]["input_ids"])
        self.assertEqual(amostrados[1]["input_ids"], exemplos[1]["input_ids"])
        self.assertEqual(diagnostico["tokens_positivos"], 2)
        self.assertEqual(diagnostico["tokens_o_usados_na_perda"], 4)
        self.assertEqual(sum(x == 0 for item in amostrados for x in item["labels"]), 4)
        self.assertTrue(any(x == 0 for x in amostrados[1]["labels"]))


class DecodificacaoRestrita(unittest.TestCase):
    def test_vazio_empates_e_o(self):
        self.assertEqual(decodificar_viterbi([]), [])
        self.assertEqual(decodificar_viterbi([[0] * 5] * 3), [0, 0, 0])
        self.assertEqual(decodificar_viterbi([[8, 0, 0, 0, 0]] * 3), [0, 0, 0])

    def test_otimo_global_confere_com_enumeracao_exaustiva(self):
        rng = random.Random(7)
        for _ in range(30):
            logits = [[rng.uniform(-5, 5) for _ in range(5)] for _ in range(4)]
            caminhos = [p for p in itertools.product(range(5), repeat=4)
                        if p[0] not in (2, 4) and all(
                            b not in (2, 4) or a in (b-1, b) for a, b in zip(p, p[1:]))]
            score = lambda p: sum(logits[i][v] for i, v in enumerate(p))
            previsto = decodificar_viterbi(logits)
            self.assertAlmostEqual(score(previsto), max(map(score, caminhos)))
            self.assertEqual(bio_para_spans(previsto, [(i, i+1) for i in range(4)])[1].invalidas, 0)

    def test_repara_inicio_e_troca_de_tipo(self):
        logits = [[0, 8, 10, 0, 0], [0, 0, 0, 8, 10]]
        self.assertEqual(decodificar_viterbi(logits), [1, 3])

    def test_viterbi_pode_rever_token_anterior(self):
        self.assertEqual(decodificar_viterbi([[3, 2, -9, -9, -9], [0, 0, 10, -9, -9]]), [1, 2])

    def test_recompoe_b_consecutivos_do_mesmo_tipo_sem_caractere_entre_offsets(self):
        # Regressão do caso real: "acórdão" tokenizado como ac|ór|dão e
        # previsto incorretamente como B-JUR, B-JUR, I-JUR.
        logits = [[0, 0, 0, 9, 0], [0, 0, 0, 9, 0], [0, 0, 0, 0, 9]]
        spans, diagnostico = bio_para_spans(
            decodificar_viterbi(logits), [(0, 2), (2, 4), (4, 7)], [.9, .7, .8]
        )
        self.assertEqual([(s.inicio, s.fim, s.score_bruto) for s in spans], [(0, 7, .7)])
        self.assertEqual(diagnostico.fragmentos, 1)

    def test_preserva_b_consecutivos_separados_por_caractere(self):
        logits = [[0, 9, 0, 0, 0], [0, 9, 0, 0, 0]]
        spans, diagnostico = bio_para_spans(
            decodificar_viterbi(logits), [(0, 2), (3, 5)]
        )
        self.assertEqual([(s.inicio, s.fim) for s in spans], [(0, 2), (3, 5)])
        self.assertEqual(diagnostico.fragmentos, 1)

    def test_viterbi_depois_da_fusao_de_janelas(self):
        off, labs, scores = combinar_logits_janelas(
            [[(0, 0), (2, 5), (6, 9)], [(6, 9), (10, 12), (0, 0)]],
            [[[0]*5, [0, 0, 0, 9, 0], [0, 0, 0, 0, 9]],
             [[0, 0, 0, 0, 7], [8, 0, 0, 0, 0], [0]*5]], decodificador='viterbi')
        self.assertEqual(off, [(2, 5), (6, 9), (10, 12)])
        self.assertEqual(labs, [3, 4, 0])
        self.assertGreater(scores[1], .99)

    def test_score_usa_rotulo_escolhido_nao_maximo(self):
        from ml_citacoes.bio import _softmax
        logits = [[[0, 8, 10, 0, 0]]]
        _, labels, scores = combinar_logits_janelas([[(0, 1)]], logits, decodificador='viterbi')
        self.assertEqual(labels, [1])
        self.assertAlmostEqual(scores[0], _softmax(logits[0][0])[1])
        self.assertLess(scores[0], .5)

    def test_rejeita_comprimentos_valores_e_modo_invalidos(self):
        casos = [([[(0, 1)]], []), ([[(0, 1)]], [[]]),
                 ([[(0, 1)]], [[[0, 1]]]), ([[(0, 1)]], [[[float('nan')]*5]])]
        for offsets, logits in casos:
            with self.assertRaises(ValueError):
                combinar_logits_janelas(offsets, logits)
        with self.assertRaises(ValueError):
            combinar_logits_janelas([], [], decodificador='errado')

    def test_inferencia_viterbi_padrao_preserva_argmax_explicito_e_ocr(self):
        from ml_citacoes.inferencia import DetectorBIO
        detector = DetectorBIO(None, None)
        texto = '⚖ Prévia\nREsp 1O50'
        inicio = texto.index('REsp')
        off = [[(inicio, len(texto))]]
        logits = [[[0, 0, 0, 8, 10]]]
        bruto = detector.decodificar(texto, off, logits, decodificador='argmax')
        self.assertEqual(detector.ultimo_diagnostico.invalidas, 1)
        restrito = detector.decodificar(texto, off, logits)
        self.assertEqual(detector.ultimo_diagnostico.invalidas, 0)
        self.assertEqual(detector.ultimo_diagnostico_bruto.invalidas, 1)
        self.assertEqual((restrito[0].inicio, restrito[0].fim), (inicio, len(texto)))
        self.assertTrue(restrito[0].transformacoes_ocr)
        self.assertLess(restrito[0].score_bruto, bruto[0].score_bruto)

    def test_detectar_usa_politica_sem_mudar_offsets_ou_requerer_torch(self):
        from unittest.mock import Mock
        from ml_citacoes.inferencia import DetectorBIO
        for modo, esperado in (('argmax', 1), ('viterbi', 0)):
            detector = DetectorBIO(None, None, decodificador=modo, temperatura=2.0)
            detector.obter_logits = Mock(return_value=([[(0, 3)]], [[[0, 0, 0, 8, 10]]]))
            spans = detector.detectar('abc')
            detector.obter_logits.assert_called_once_with('abc')
            self.assertEqual(detector.ultimo_diagnostico.invalidas, esperado)
            self.assertEqual((spans[0].inicio, spans[0].fim, spans[0].tipo), (0, 3, 'JUR'))
            self.assertIsNotNone(spans[0].score_calibrado)
        with self.assertRaises(ValueError):
            DetectorBIO(None, None, decodificador='crf')


class ExperimentoPareado(unittest.TestCase):
    def preparar(self, raiz):
        dados, execucao = raiz / 'dados', raiz / 'execucao'
        dados.mkdir()
        execucao.mkdir()
        docs = [
            {'documento_id': 'treino', 'texto': 'Texto de treino.', 'spans': [], 'origem': 'original'},
            {'documento_id': 'validacao', 'texto': 'Referência inventada', 'nivel': '2', 'origem': 'original',
             'spans': [{'inicio': 0, 'fim': 10, 'tipo': 'lei', 'trecho': 'Referência'}]}]
        self.gravar_dados(dados, docs)
        (dados / 'folds.json').write_text(json.dumps({'folds': [
            {'fold': 0, 'treino': ['treino'], 'validacao': ['validacao']}
        ]}))
        manifesto = {'fold': '0', 'documentos_treino': ['treino'], 'documentos_validacao': ['validacao'],
                     'dataset_sha256': hashlib.sha256((dados / 'documentos.jsonl').read_bytes()).hexdigest()}
        (execucao / 'manifesto.json').write_text(json.dumps(manifesto))
        return dados, execucao, docs, manifesto

    def gravar_dados(self, dados, docs):
        (dados / 'documentos.jsonl').write_text('\n'.join(json.dumps(d) for d in docs))

    def test_smoke_sem_torch_uma_passagem_mesmos_logits(self):
        from ml_citacoes.experimento_bio import executar
        class FalsoDetector:
            chamadas = 0
            def obter_logits(self, texto):
                self.chamadas += 1
                return [[(0, 10), (11, 20)]], [[[0, 8, 10, 0, 0], [10, 0, 0, 0, 0]]]
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            dados, execucao, _, _ = self.preparar(raiz)
            args = SimpleNamespace(dados=dados, execucao=execucao, saida=raiz/'saida', limiar_lei=.99, limiar_jur=.99)
            detector = FalsoDetector()
            r = executar(args, detector)
            self.assertEqual(detector.chamadas, 1)
            self.assertEqual(r['variantes']['argmax']['diagnostico_bio']['documentos_invalidos'], 1)
            self.assertEqual(r['variantes']['viterbi']['diagnostico_bio']['documentos_invalidos'], 0)
            self.assertEqual(r['variantes']['viterbi']['bruto']['iou_05']['geral']['tp'], 1)
            self.assertEqual(r['variantes']['viterbi']['limiar']['iou_05']['geral']['fn'], 1)
            self.assertIn('2', r['variantes']['viterbi']['bruto']['por_nivel'])
            self.assertIsNone(r['estabilidade_entre_folds']['desvio'])
            resumo = (args.saida / 'resumo.json').read_text(encoding='utf-8')
            self.assertNotIn('Referência', resumo)
            self.assertNotIn('documento_id', resumo)
            self.assertTrue((args.saida / 'erros_locais.json').exists())
            with self.assertRaisesRegex(ValueError, 'saída já existe'):
                executar(args, detector)

    def test_rejeita_fold_diferente_hash_e_vazamento(self):
        from ml_citacoes.experimento_bio import validar_entrada
        with tempfile.TemporaryDirectory() as tmp:
            dados, execucao, docs, m = self.preparar(Path(tmp))
            for campo, valor, mensagem in [('fold', '1', 'fold da execução ausente'),
                                           ('dataset_sha256', 'errado', 'dataset diverge'),
                                           ('documentos_treino', ['validacao'], 'vazamento')]:
                (execucao/'manifesto.json').write_text(json.dumps({**m, campo: valor}))
                with self.assertRaisesRegex(ValueError, mensagem):
                    validar_entrada(dados, execucao)
            docs[0]['texto'] = docs[1]['texto']
            self.gravar_dados(dados, docs)
            m['dataset_sha256'] = hashlib.sha256((dados/'documentos.jsonl').read_bytes()).hexdigest()
            (execucao/'manifesto.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'texto idêntico'):
                validar_entrada(dados, execucao)

    def test_regras_recall_perfeito_nao_mascara_fp_adicionado(self):
        from unittest.mock import patch
        from ml_citacoes.experimento_bio import executar
        class FalsoDetector:
            def obter_logits(self, texto):
                return [[(0, 1)]], [[[0, 10, 0, 0, 0]]]
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            dados, execucao, _, _ = self.preparar(raiz)
            args = SimpleNamespace(dados=dados, execucao=execucao, saida=raiz/'saida', limiar_lei=.99, limiar_jur=.99)
            regra = {'inicio': 0, 'fim': 10, 'tipo': 'lei', 'trecho': 'Referência'}
            with patch('ml_citacoes.experimento_bio.extrair', return_value=[regra]), \
                 patch('ml_citacoes.experimento_bio.filtrar_distratores', side_effect=lambda rs, texto: rs), \
                 patch('ml_citacoes.experimento_bio.e_distrator', return_value=False):
                r = executar(args, FalsoDetector())
            self.assertTrue(r['regras_perfeitas_em_recall'])
            for modo in ('argmax', 'viterbi'):
                self.assertEqual(r['variantes'][modo]['recall_adicional_sobre_regras'], 0)
                self.assertEqual(r['variantes'][modo]['fp_realmente_acrescentados'], 1)
                self.assertEqual(r['variantes'][modo]['exclusivos_adicionados']['geral']['precisao'], 0)

    def test_fragmentacao_compara_somente_mesmo_documento(self):
        from ml_citacoes.experimento_bio import _fragmentacao
        verdade = [{'documento_id': 'a', 'inicio': 0, 'fim': 10, 'tipo': 'LEI'}]
        previsto = [{'documento_id': 'b', 'inicio': 0, 'fim': 4, 'tipo': 'LEI'},
                    {'documento_id': 'a', 'inicio': 5, 'fim': 10, 'tipo': 'LEI'}]
        self.assertEqual(_fragmentacao(verdade, previsto)['taxa'], 0)
        previsto[0]['documento_id'] = 'a'
        self.assertEqual(_fragmentacao(verdade, previsto)['taxa'], 1)

    def test_experimento_verificador_reporta_funil_sem_trechos_no_resumo(self):
        from unittest.mock import patch
        from ml_citacoes.bio import DiagnosticoBIO
        from ml_citacoes.experimento_verificador import executar

        class Detector:
            ultimo_diagnostico = DiagnosticoBIO()

            def detectar(self, texto):
                return [CitacaoModelo(0, 10, 'LEI', .2)]

        class Verificador:
            def classificar_muitos(self, trechos):
                return [ResultadoVerificacao('CITACAO_LEGISLACAO', .7, {
                    'CITACAO_LEGISLACAO': .7, 'CITACAO_JURISPRUDENCIA': .1,
                    'INCERTO': .1, 'NAO_CITACAO': .1,
                }, 'falso', 'hash') for _ in trechos]

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            dados, execucao, _, _ = self.preparar(raiz)
            args = SimpleNamespace(dados=dados, execucao=execucao, saida=raiz/'saida')
            with patch('ml_citacoes.experimento_verificador.extrair', return_value=[]):
                resultado = executar(args, Detector(), Verificador())
            self.assertEqual(resultado['verificador']['contagens']['CITACAO_LEGISLACAO'], 1)
            self.assertEqual(resultado['recall_adicional_aceito_sobre_regras'], 1.0)
            self.assertEqual(resultado['fp_realmente_acrescentados'], 0)
            resumo = (args.saida/'resumo.json').read_text(encoding='utf-8')
            self.assertNotIn('Referência', resumo)
            detalhes = json.loads((args.saida/'erros_locais.json').read_text(encoding='utf-8'))
            self.assertEqual(detalhes['decisoes_llm'][0]['decisao'], 'CITACAO_LEGISLACAO')
            self.assertEqual(detalhes['decisoes_llm'][0]['trecho'], 'Referência')

    def test_agregacao_exige_cinco_folds_compativeis_e_reproduzidos(self):
        from ml_citacoes.agregar_experimento_bio import agregar

        def resumo(fold):
            metrica = {
                'geral': {'tp': 10, 'fp': 2, 'fn': 1, 'precisao': .8, 'recall': .9, 'f1': .85},
                'por_tipo': {
                    'LEI': {'tp': 2, 'fp': 1, 'fn': 0, 'precisao': .7, 'recall': 1., 'f1': .8},
                    'JUR': {'tp': 8, 'fp': 1, 'fn': 1, 'precisao': .9, 'recall': .8, 'f1': .85},
                },
            }
            grupo = {'iou_05': metrica, 'exato': metrica}
            variantes = {}
            for modo in ('argmax', 'viterbi'):
                variantes[modo] = {
                    'bruto': grupo, 'limiar': grupo, 'fusao': grupo,
                    'diagnostico_bio': {
                        'transicoes_invalidas': int(modo == 'argmax'),
                        'documentos_invalidos': int(modo == 'argmax'),
                        'taxa_documentos_invalidos': .2 if modo == 'argmax' else 0.,
                    },
                    'colapso': False,
                }
            return {
                'fold': fold, 'dataset_sha256': 'dados', 'codigo_sha256': {'a': '1'},
                'regras_config_sha256': {'r': '1'}, 'limiares_fixos': {'LEI': .99, 'JUR': .99},
                'calibracao': 'nenhuma', 'ocr_definicao': 'nivel2', 'n_documentos': 2,
                'n_entidades': 11, 'tokens_alterados': fold, 'regras_perfeitas_em_recall': True,
                'reproducao_argmax': {'documentos_com_spans_diferentes': 0},
                'variantes': variantes,
                'delta_viterbi_menos_argmax': {
                    nome: {k: 0 for k in ('tp', 'fp', 'fn', 'precisao', 'recall', 'f1')}
                    for nome in ('bruto', 'limiar', 'fusao')
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            caminhos = []
            for fold in range(5):
                caminho = raiz / f'{fold}.json'
                caminho.write_text(json.dumps(resumo(fold)))
                caminhos.append(caminho)
            resultado = agregar(caminhos)
            self.assertEqual(resultado['folds'], [0, 1, 2, 3, 4])
            self.assertEqual(resultado['tokens_alterados']['total'], 10)
            self.assertEqual(resultado['variantes']['argmax']['bruto']['iou_05']['totais']['fp'], 10)
            self.assertEqual(resultado['variantes']['viterbi']['diagnostico_bio']['transicoes_invalidas_total'], 0)
            with self.assertRaisesRegex(ValueError, 'exatamente cinco'):
                agregar(caminhos[:4])
            adulterado = resumo(4)
            adulterado['codigo_sha256'] = {'a': 'outro'}
            caminhos[4].write_text(json.dumps(adulterado))
            with self.assertRaisesRegex(ValueError, 'codigo_sha256 divergente'):
                agregar(caminhos)


class IntegracaoFontePipeline(unittest.TestCase):
    def verificar(self, usar_regra, usar_modelo, fonte):
        from unittest.mock import Mock, patch
        from pipeline import processar_documento
        texto = 'art. 5º da CF'
        regra = {'tipo': 'lei', 'inicio': 0, 'fim': len(texto), 'trecho': texto}
        detector = Mock() if usar_modelo else None
        if detector is not None:
            detector.decodificador = 'viterbi'
            detector.detectar.return_value = [CitacaoModelo(0, len(texto), 'LEI', .999)]
        resolvedor = Mock()
        resolvedor.resolver_citacao.return_value = SimpleNamespace(
            caminho='teste', classe='real', id_canonico=None, cobertura='fora_base', motivo='teste')
        with patch('pipeline.extrair', return_value=[regra] if usar_regra else []), \
             patch('pipeline.filtrar_distratores', side_effect=lambda rs, texto: rs), \
             patch('pipeline.e_distrator', return_value=False), \
             patch('pipeline.atribuir_confianca', return_value=.8) as confianca:
            resultado = processar_documento('sintetico', texto, resolvedor, detector)
        self.assertEqual(len(resultado['citacoes']), 1)
        self.assertEqual(resultado['citacoes'][0]['fonte'], fonte)
        if usar_modelo:
            self.assertEqual(resultado['decodificador_bio'], 'viterbi')
        self.assertEqual(resolvedor.resolver_citacao.call_args.args[0].fonte, fonte)
        confianca.assert_called_once_with('teste', fonte)
        return resultado

    def test_exclusivo_modelo_propaga_fonte_ate_confianca(self):
        r = self.verificar(False, True, 'modelo')
        self.assertEqual(r['citacoes'][0]['evidencia_deteccao']['origem'], 'modelo')

    def test_concordancia_preserva_fonte_regra_e_evidencia_ambos(self):
        r = self.verificar(True, True, 'regra')
        self.assertEqual(r['citacoes'][0]['evidencia_deteccao']['origem'], 'ambos')

    def test_sem_modelo_preserva_novo_contrato_remoto(self):
        r = self.verificar(True, False, 'regra')
        self.assertNotIn('eventos_deteccao', r)


if __name__ == "__main__":
    unittest.main()
