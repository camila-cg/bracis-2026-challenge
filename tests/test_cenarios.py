"""Regressões do resolvedor e pendências conhecidas do pipeline.

Execute da raiz: python -m unittest discover -s tests -v
"""

import csv
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import indice_utils as ind
from modelos import CitacaoNormalizada
from steps.buscar_na_base import Resolvedor
from steps.atribuir_confianca import atribuir_confianca
from steps.normalizar_citacao import normalizar_citacao
from steps.normalizar_numeros import normalizar_trecho
from steps.pipeline_utils import BLOCO_NUMERICO
from steps.salvar_submissao import gravar_saida


class ResolucaoAcordaos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        banco = RAIZ / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        cls.con = sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True)
        cls.resolvedor = Resolvedor(cls.con)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "con"):
            cls.con.close()

    def test_citacao_correta_resolve(self):
        self.assertEqual(self.resolvedor.acordao("REsp 1.741.784/PR")[:2], ("real", 2566535283))

    def test_tribunal_divergente_nao_resolve(self):
        self.assertEqual(self.resolvedor.acordao("RE 1.741.784 do STF")[:2], ("inventada", None))

    def test_uf_divergente_nao_resolve(self):
        self.assertEqual(self.resolvedor.acordao("REsp 1.741.784/SC")[:2], ("inventada", None))

    def test_classe_divergente_nao_resolve(self):
        self.assertEqual(self.resolvedor.acordao("AREsp 1.741.784/PR")[:2], ("inventada", None))

    def test_fallback_fts_tambem_filtra_metadados(self):
        with patch.object(ind, "resolver", return_value=set()):
            self.assertEqual(self.resolvedor.acordao("REsp 1.741.784/SC")[:2], ("inventada", None))

    def test_estagio_recursal_desempata_colisao_de_numero(self):
        # 1.599.372/PR tem dois acórdãos reais no mesmo tribunal/UF/classe
        # (mesmo balde "recursoespecial"): um AgInt, outro Embargos de
        # Divergência. Sem o estágio, fica ambíguo — com o estágio citado
        # explicitamente, desempata para o acórdão certo.
        self.assertEqual(self.resolvedor.acordao("AgInt no REsp 1.599.372/PR")[:2], ("real", 2684934117))
        self.assertEqual(
            self.resolvedor.acordao("Embargos de Divergência no REsp 1.599.372/PR")[:2],
            ("real", 2679381856),
        )
        self.assertEqual(self.resolvedor.acordao("REsp 1.599.372/PR")[:2], ("incompleta", None))

    def test_estagio_recursal_ausente_do_cabecalho_nao_zera_candidatos(self):
        # 36.670/SP é um acórdão de AgInt cujo cabeçalho registra só a
        # classe de base ("RECLAMAÇÃO", sem "AgInt"). O desempate por
        # estágio não pode filtrar isso a zero — precisa cair pro
        # comportamento anterior (resolve pelo número, único candidato).
        self.assertEqual(self.resolvedor.acordao("AgInt na Rcl 36.670/SP")[:2], ("real", 2679568224))

    def test_apelacao_generica_resolve_registro_de_apelacao_criminal(self):
        # A citação diz só "APL" (genérico); o acórdão real no STM tem
        # classe "APELAÇÃO CRIMINAL" (mais específica). Exigir bater
        # exatamente zerava candidatos reais — a citação nunca teve como
        # informar que era criminal.
        self.assertEqual(
            self.resolvedor.acordao("APL 7000449-40.2023.7.00.0000/RS")[:2],
            ("real", 5665364632),
        )


class ResolucaoNormativos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        banco = RAIZ / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        cls.con = sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True)
        cls.resolvedor = Resolvedor(cls.con)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "con"):
            cls.con.close()

    def test_artigo_sem_estatuto_deveria_ser_incompleto(self):
        resultado = self.resolvedor.lei("art. 373")
        self.assertEqual(resultado, ("incompleta", None, "lei_sem_estatuto"))
        self.assertEqual(atribuir_confianca(resultado[2]), 0.60)

    def test_sumula_sem_tribunal_deveria_ser_incompleta(self):
        resultado = self.resolvedor.sumula("Súmula 83")
        self.assertEqual(resultado, ("incompleta", None, "sumula_sem_tribunal"))
        self.assertEqual(atribuir_confianca(resultado[2]), 0.60)

    def test_artigo_com_estatuto_continua_real_ou_inventado(self):
        self.assertEqual(self.resolvedor.lei("art. 373 do CPC")[:2], ("real", 28893055))
        self.assertEqual(self.resolvedor.lei("art. 999 do CPC")[:2], ("inventada", None))

    def test_lei_numerada_fora_da_base_continua_inventada(self):
        self.assertEqual(self.resolvedor.lei("art. 172 da Lei nº 9.504/1997")[:2], ("inventada", None))
        self.assertEqual(self.resolvedor.lei("art. 60 da Lei nº 13.467/2017")[:2], ("inventada", None))

    def test_sumula_com_tribunal_continua_real_ou_inventada(self):
        self.assertEqual(self.resolvedor.sumula("Súmula 83 do STJ")[:2], ("real", 1289710642))
        self.assertEqual(self.resolvedor.sumula("Súmula 999 do STJ")[:2], ("inventada", None))

    def test_sumula_vinculante_identifica_stf_sem_sufixo(self):
        self.assertEqual(self.resolvedor.sumula("Súmula Vinculante 10")[:2], ("real", 1289712966))


class NormalizacaoDeIdentificadores(unittest.TestCase):
    def test_ano_separado_nao_e_unido_ao_numero_do_processo(self):
        for separador in (" - ", " – ", " \n- "):
            with self.subTest(separador=repr(separador)):
                trecho = f"REsp 1.741.784{separador}2024"
                self.assertEqual(
                    [m.group(0) for m in BLOCO_NUMERICO.finditer(trecho)],
                    ["1.741.784", "2024"],
                )
                self.assertEqual(
                    normalizar_trecho(trecho),
                    f"REsp 1.741.784{separador}2.024",
                )

    def test_separadores_quebrados_dentro_do_identificador_continuam_unidos(self):
        self.assertEqual(normalizar_trecho("REsp 1.741. 784/PR"), "REsp 1.741.784/PR")
        self.assertEqual(normalizar_trecho("REsp 33.-\n474/PR"), "REsp 33.474/PR")
        self.assertEqual(
            [m.group(0) for m in BLOCO_NUMERICO.finditer("7220273--\n23.2018")],
            ["7220273--\n23.2018"],
        )

    def test_ano_separado_preserva_resolucao_de_acordao(self):
        banco = RAIZ / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            self.skipTest("base canônica ausente em data/desafio1_bracis.db")
        with sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True) as con:
            resolvedor = Resolvedor(con)
            trecho = normalizar_trecho("REsp 1.741.784 - 2024/PR")
            self.assertEqual(resolvedor.acordao(trecho)[:2], ("real", 2566535283))


class FonteDaCitacao(unittest.TestCase):
    """Hook pra distinguir citações vindas das regras fixas vs. de um
    extrator alternativo (ex. o modelo generalizável da Giovanna). O desconto
    é 0.0 hoje de propósito — o modelo ainda não existe pra calibrar — então
    isto documenta que o hook não muda nada até alguém ajustar o desconto.
    """

    def test_citacao_normalizada_sem_fonte_explicita_assume_regra(self):
        candidato = {"trecho": "REsp 1.741.784/PR", "inicio": 0, "fim": 18}
        citacao = normalizar_citacao(candidato, "acordao", ja_normalizado=True)
        self.assertEqual(citacao.fonte, "regra")

    def test_citacao_normalizada_propaga_fonte_do_candidato(self):
        candidato = {"trecho": "REsp 1.741.784/PR", "inicio": 0, "fim": 18, "fonte": "modelo"}
        citacao = normalizar_citacao(candidato, "acordao", ja_normalizado=True)
        self.assertEqual(citacao.fonte, "modelo")

    def test_fonte_modelo_ainda_e_placeholder_neutro(self):
        # Quando o desconto for calibrado com dados reais do modelo, este
        # teste precisa ser atualizado de propósito — não é pra continuar
        # passando sem mais ninguém perceber a mudança de comportamento.
        base = atribuir_confianca("lei_real")
        self.assertEqual(atribuir_confianca("lei_real", "regra"), base)
        self.assertEqual(atribuir_confianca("lei_real", "modelo"), base)

    def test_desconto_fonte_modelo_nunca_sai_do_intervalo_0_1(self):
        # Um desconto negativo vira bônus (1 - desconto > 1) e um desconto
        # maior que 1 vira negativo — os dois estourariam [0, 1], o que a
        # métrica do desafio rejeita pra submissão inteira. O resultado tem
        # que ficar sempre dentro do intervalo, não importa o desconto.
        with patch("steps.atribuir_confianca.DESCONTO_FONTE_MODELO", -0.5):
            self.assertEqual(atribuir_confianca("acordao_indice", "modelo"), 1.0)
        with patch("steps.atribuir_confianca.DESCONTO_FONTE_MODELO", 1.5):
            self.assertEqual(atribuir_confianca("acordao_indice", "modelo"), 0.0)


class SaidaDaSubmissao(unittest.TestCase):
    def test_segunda_execucao_nao_deveria_incluir_json_antigo(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta)
            with patch.object(sys, "argv", ["pipeline.py"]):
                gravar_saida([{"documento_id": "anterior", "citacoes": []}], destino)
                csv_saida = gravar_saida([{"documento_id": "atual", "citacoes": []}], destino)
                self.assertEqual(sys.argv, ["pipeline.py"])
            with csv_saida.open(encoding="utf-8", newline="") as arquivo:
                ids = [linha["documento_id"] for linha in csv.DictReader(arquivo)]
            self.assertEqual(ids, ["atual"])
            self.assertTrue((destino / "json" / "anterior.json").is_file())
            self.assertTrue((destino / "json" / "atual.json").is_file())

    def test_csv_preserva_formato_do_conversor(self):
        citacao = {
            "inicio": 10, "fim": 25, "classificacao": "real",
            "resolucao": {"id_canonico": "123"}, "confianca": 0.9,
        }
        with tempfile.TemporaryDirectory() as pasta:
            csv_saida = gravar_saida([
                {"documento_id": "com_citacao", "citacoes": [citacao]},
                {"documento_id": "sem_citacao", "citacoes": []},
            ], Path(pasta))
            with csv_saida.open(encoding="utf-8", newline="") as arquivo:
                linhas = list(csv.DictReader(arquivo))
        self.assertEqual(linhas, [
            {"documento_id": "com_citacao", "citacoes": "10,25,real,123,0.9000"},
            {"documento_id": "sem_citacao", "citacoes": "-"},
        ])


if __name__ == "__main__":
    unittest.main()
