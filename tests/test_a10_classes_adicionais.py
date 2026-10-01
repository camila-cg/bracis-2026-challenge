"""A10: classes canônicas raras que ainda não iniciavam extração."""

import sqlite3
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor


# (citação, classe esperada, id canônico); todas vêm da base do desafio.
CASOS_REAIS = (
    ("Ação Penal nº 1.140/DF do STJ", "Ação Penal", 6621798657),
    ("Ação Direta de Inconstitucionalidade nº 2.110/DF do STF", "Ação Direta", 2794008744),
    ("Ação Rescisória nº 5.446/PR do STJ", "Ação Rescisória", 3582380541),
    ("Suspensão de Segurança nº 3.078/SP do STJ", "Suspensão de Segurança", 1882417791),
    ("Suspensão de Liminar e de Sentença nº 2.883/MA do STJ", "Suspensão de Liminar", 6576893354),
    ("Petição nº 12.344/DF do STJ", "Petição", 2106307633),
    ("Prestação de Contas nº 0000901-76.2011.6.00.0000 do TSE", "Prestação de Contas", 349613640),
    ("Ag-ROT nº 1005710-02.2023.5.02.0000 do TST", "Ag-ROT", 2676113914),
    ("EDCiv-ROT nº 0165500-53.1989.5.03.0016 do TST", "EDCiv-ROT", 2858650868),
    ("IRR nº 0000849-83.2013.5.03.0138 do TST", "IRR", 1974903203),
    ("Conflito de Jurisdição nº 7000686-16.2019.7.00.0000 do STM", "Conflito de Jurisdição", 2381366854),
    ("Incompatibilidade nº 7000804-84.2022.7.00.0000/DF do STM", "Incompatibilidade", 2382427517),
    ("Correição Parcial Militar nº 7000503-35.2025.7.00.0000/RJ do STM", "Correição Parcial Militar", 5098795910),
)


class ClassesAdicionais(unittest.TestCase):
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

    def processar(self, citacao: str) -> list[dict]:
        texto = f"Nesse sentido, confira-se {citacao}, cujo entendimento se aplica."
        return processar_documento("a10", texto, self.resolvedor)["citacoes"]

    def test_classes_raras_resolvem_o_id_canonico(self):
        for citacao, inicio_span, id_esperado in CASOS_REAIS:
            with self.subTest(citacao=citacao):
                encontradas = self.processar(citacao)
                self.assertEqual(
                    len(encontradas), 1,
                    f"citações detectadas: {[item['trecho'] for item in encontradas]}",
                )
                resultado = encontradas[0]
                self.assertTrue(resultado["trecho"].startswith(inicio_span))
                self.assertEqual(resultado["classificacao"], "real")
                self.assertEqual(
                    resultado["resolucao"], {"id_canonico": str(id_esperado)}
                )

    def test_lista_triplice_duplicada_permanece_incompleta(self):
        encontradas = self.processar(
            "Lista Tríplice nº 0600221-90.2021.6.00.0000 do TSE"
        )
        self.assertEqual(len(encontradas), 1)
        self.assertEqual(encontradas[0]["classificacao"], "incompleta")

    def test_modificador_nao_substitui_a_classe_base(self):
        encontradas = self.processar(
            "Agravo Interno na Suspensão de Liminar e de Sentença nº 2.883/MA do STJ"
        )
        self.assertEqual(len(encontradas), 1)
        self.assertEqual(encontradas[0]["classificacao"], "real")
        self.assertEqual(
            encontradas[0]["resolucao"], {"id_canonico": "6576893354"}
        )

    def test_palavras_sem_identificador_processual_nao_viram_citacao(self):
        texto = (
            "A incompatibilidade foi corrigida. A petição será protocolada. "
            "A prestação de contas está em elaboração e a lista tríplice foi publicada."
        )
        self.assertEqual(
            processar_documento("controles", texto, self.resolvedor)["citacoes"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
