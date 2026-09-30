"""Abreviações e formas que o extrator não reconhecia ou cortava.

Cobre `C L T`/`C P C` (letras separadas), `Enunciado`, `Decreto`/`Decr.`/`Res.`/`Resolução`
(atos normativos fora do catálogo), o diploma antes do artigo (`CF, art. 93, IX`,
`Cód. de Proc. Penal-Militar, art. 5º`) e `arts.` no plural.

O dev oficial não tem nenhuma dessas formas (só nomes por extenso e as siglas
`CLT`/`CPC`/`CPP`): elas vêm dos conjuntos sintéticos. Por isso os testes incluem
guardas contra falso positivo em prosa comum.

`IAC` ("QO no IAC 15/STJ") ficou de fora: o span invade a frase seguinte e piora o
gate de distratores; depende de corrigir a fronteira de span.
"""

import sqlite3
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor

MOLDURA = "Conforme {}, decidiu-se.\n"

ID_CLT_896 = 10637358
ID_CPC_373 = 28893055
ID_CF_93 = 10626510
ID_SUMULA_TST_331 = 1431369957


class BaseCanonica(unittest.TestCase):
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

    def citacoes(self, citacao: str) -> list[dict]:
        return processar_documento("sintetico", MOLDURA.format(citacao), self.resolvedor)["citacoes"]

    def uma(self, citacao: str) -> dict:
        achadas = self.citacoes(citacao)
        self.assertEqual(len(achadas), 1, f"citações detectadas: {[c['trecho'] for c in achadas]}")
        return achadas[0]


class LetrasSeparadas(BaseCanonica):
    def test_clt_e_cpc_com_letras_separadas_entram_no_span_e_resolvem(self):
        for texto, sigla, id_esperado in (
            ("o art. 896 da C L T", "C L T", ID_CLT_896),
            ("o art. 373, § 1º, do C P C", "C P C", ID_CPC_373),
        ):
            with self.subTest(texto=texto):
                c = self.uma(texto)
                self.assertIn(sigla, c["trecho"])
                self.assertEqual(c["classificacao"], "real")
                self.assertEqual(c["resolucao"], {"id_canonico": str(id_esperado)})

    def test_artigo_inexistente_com_letras_separadas_continua_inventada(self):
        self.assertEqual(self.uma("o art. 99999 da C L T")["classificacao"], "inventada")


class Enunciado(BaseCanonica):
    def test_enunciado_com_tribunal_e_como_sumula(self):
        c = self.uma("o Enunciado 331 do TST")
        self.assertTrue(c["trecho"].startswith("Enunciado"), c["trecho"])
        self.assertEqual(c["classificacao"], "real")
        self.assertEqual(c["resolucao"], {"id_canonico": str(ID_SUMULA_TST_331)})

    def test_enunciado_inexistente_continua_inventada(self):
        self.assertEqual(self.uma("o Enunciado 99999 do TST")["classificacao"], "inventada")

    def test_enunciado_de_orgao_que_nao_e_tribunal_e_inventado(self):
        # Decisão adotada como padrão (ainda a confirmar): enunciado sem tribunal conhecido não está no
        # catálogo; identificador explícito sem mapeamento é inventado, como a lei numerada.
        for texto in ("o Enunciado 92 do Fórum Nacional da Infância, aprovado em 2022", "o Enunciado 92"):
            with self.subTest(texto=texto):
                c = self.uma(texto)
                self.assertTrue(c["trecho"].startswith("Enunciado"), c["trecho"])
                self.assertEqual(c["classificacao"], "inventada")
                self.assertEqual(c["motivo"], "enunciado_sem_mapeamento")

    def test_sumula_sem_tribunal_continua_incompleta(self):
        # A regra vale só para `Enunciado`: a súmula sem tribunal segue como antes.
        self.assertEqual(self.uma("a Súmula 83")["classificacao"], "incompleta")


class AtosNormativosForaDoCatalogo(BaseCanonica):
    def test_decreto_e_resolucao_numerados_entram_no_span_e_sao_inventadas(self):
        for texto, prefixo in (
            ("o Decreto 10.000/2019, art. 2", "Decreto"),
            ("a Res. CNJ nº 455, de 27-4-2022", "Res."),
            ("a Resolução CNJ 400/2021, art. 2", "Resolução"),
            ("o Decr. 9.83O, de 1O/6/2O19", "Decr."),
        ):
            with self.subTest(texto=texto):
                c = self.uma(texto)
                self.assertTrue(c["trecho"].startswith(prefixo), c["trecho"])
                self.assertEqual(c["classificacao"], "inventada")

    def test_decreto_lei_do_catalogo_continua_resolvendo(self):
        c = self.uma("o art. 896 do Decreto-Lei nº 5.452/1943")
        self.assertEqual(c["classificacao"], "real")
        self.assertEqual(c["resolucao"], {"id_canonico": str(ID_CLT_896)})


class DiplomaAntesDoArtigo(BaseCanonica):
    def test_cf_antes_do_artigo_entra_no_span_e_resolve(self):
        c = self.uma("a CF, art. 93, IX")
        self.assertTrue(c["trecho"].startswith("CF"), c["trecho"])
        self.assertEqual(c["classificacao"], "real")
        self.assertEqual(c["resolucao"], {"id_canonico": str(ID_CF_93)})

    def test_codigo_abreviado_antes_do_artigo_entra_no_span(self):
        c = self.uma("o Cód. de Proc. Penal-Militar, art. 5º, inc. II")
        self.assertTrue(c["trecho"].startswith("Cód."), c["trecho"])
        self.assertEqual(c["classificacao"], "inventada")


class ArtigosNoPlural(BaseCanonica):
    def test_arts_com_quebra_de_linha_e_detectado_com_o_diploma(self):
        c = self.uma("os arts. 5º, LIV e LV, da\nConstituição Federal de 1988")
        self.assertTrue(c["trecho"].startswith("arts."), c["trecho"])
        self.assertIn("Constituição Federal", c["trecho"])


class GuardasContraFalsoPositivo(BaseCanonica):
    def test_palavras_comuns_em_prosa_nao_viram_citacao(self):
        # Limite conhecido (falsos positivos em prosa): marcador de lei + número solto em prosa ("a lei anterior
        # 5 dias depois") vira citação `incompleta`, com `lei` e agora também com `decreto`/`resolução`.
        for texto in (
            "O enunciado da questão foi revisado pela banca examinadora.\n",
            "O decreto foi publicado ontem no Diário Oficial da União.\n",
            "A res judicata impede o ajuizamento de nova ação idêntica.\n",
            "A resolução do problema exige paciência e método.\n",
            "Os arts. estão numerados no final da página.\n",
            "A sigla CF aparece em várias planilhas internas.\n",
            "O código de conduta da empresa foi atualizado.\n",
            "O cód. de barras 7891234567890 consta no rótulo do produto.\n",
            "O Proc. 123 tramita na vara cível da comarca.\n",
            "Cf. fls. 12, o réu foi citado por edital.\n",
            "Segundo o CF desta planilha, o saldo final é 12.\n",
            "O decreto presidencial foi revogado cinco dias depois da publicação.\n",
        ):
            with self.subTest(texto=texto.strip()):
                citacoes = processar_documento("sintetico", texto, self.resolvedor)["citacoes"]
                self.assertEqual([c["trecho"] for c in citacoes], [])


if __name__ == "__main__":
    unittest.main()
