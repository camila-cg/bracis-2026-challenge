"""Acórdãos que o parser de cabeçalho não lia (TSE com encoding quebrado, TST e STM)."""

import sqlite3
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

import indice_utils as ind


class ParserSintetico(unittest.TestCase):
    def test_tse_com_encoding_quebrado_le_classe_e_cnj_completo(self):
        texto = (
            "TRIBUNAL SUPERIOR ELEITORAL AC€RD O RECURSO ESPECIAL ELEITORAL "
            "N‚ 184-70.2012.6.20.0033 -CLASSE 32ƒ MOSSOR€ - RIO GRANDE DO NORTE Relatora"
        )
        campos = ind.parsear_cabecalho("TSE", texto)
        self.assertEqual(campos["classe"], "RECURSO ESPECIAL ELEITORAL")
        self.assertEqual(ind.normalizar_numero(campos["numero"]), "0000184-7020126200033")
        self.assertEqual(ind.normalizar_uf(campos["uf"]), "RN")

    def test_tse_com_espaco_apos_o_ponto_do_cnj(self):
        texto = (
            "TRIBUNAL SUPERIOR ELEITORAL AC€RD O AGRAVO REGIMENTAL NO RECURSO ESPECIAL "
            "ELEITORAL N‚ 374-42. 2012.6.16.0066 - CLASSE 32ƒ MARING„"
        )
        campos = ind.parsear_cabecalho("TSE", texto)
        self.assertEqual(campos["classe"], "AGRAVO REGIMENTAL NO RECURSO ESPECIAL ELEITORAL")
        self.assertEqual(ind.normalizar_numero(campos["numero"]), "0000374-4220126160066")
        self.assertIsNone(ind.normalizar_uf(campos["uf"]))  # "PARAN„": acento quebrado

    def test_tse_sem_cnj_completo_continua_sem_parser(self):
        texto = (
            "TRIBUNAL SUPERIOR ELEITORAL AC€RD O RECURSO ESPECIAL ELEITORAL "
            "N‚ 185-05. 20.16.6. , 250024 - CLASSE 6 SERGIPE Relator"
        )
        self.assertIsNone(ind.parsear_cabecalho("TSE", texto))

    def test_tst_formula_com_ruido_antes_do_codigo(self):
        for marcador in ("n.º", "n . º", "nº", "n.�"):
            with self.subTest(marcador=marcador):
                texto = (
                    f"Vistos, relatados e discutidos estes autos de Agravo de Instrumento "
                    f"em Recurso de Revista {marcador} TST-AIRR-88600-13.2008.5.24.0005 , em que"
                )
                campos = ind.parsear_cabecalho("TST", texto)
                self.assertEqual(campos["classe"], "AIRR")
                self.assertEqual(campos["numero"], "88600-13.2008.5.24.0005")

    def test_tst_usa_a_formula_e_nao_o_precedente_citado_depois(self):
        texto = (
            "autos de Agravo n . º TST-Ag-ED-AIRR-155000-36.2009.5.01.0030 , em que. "
            "Mais adiante: Incidente nº TST-IncJulgRREmbRep-1000-71.2012.5.06.0018, fixou."
        )
        campos = ind.parsear_cabecalho("TST", texto)
        self.assertEqual(campos["classe"], "Ag-ED-AIRR")
        self.assertEqual(campos["numero"], "155000-36.2009.5.01.0030")

    def test_stm_numero_curto_depois_da_ata_exige_repeticao(self):
        ata = "EXTRATO DA ATA " + "Ministros presentes. " * 60
        titulo = "APELAÇÃO Nº 234-73.2016.7.11.0211 - DF - Relator Ministro X. "
        pagina = "SUPERIOR TRIBUNAL MILITAR APELAÇÃO Nº 234-73.2016.7.11.0211/DF "
        repetido = ata + titulo + pagina * 2
        campos = ind.parsear_cabecalho("STM", repetido)
        self.assertEqual(campos["classe"], "APELAÇÃO")
        self.assertEqual(ind.normalizar_numero(campos["numero"]), "0000234-7320167110211")
        self.assertEqual(campos["uf"], "DF")
        # citado uma única vez: não é o processo do documento
        self.assertIsNone(ind.parsear_cabecalho("STM", ata + titulo))


class BaseCanonica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        banco = RAIZ / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        cls.con = sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True)
        cls.indice, cls.sem_parser = ind.construir_indice_acordaos(cls.con)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "con"):
            cls.con.close()

    def test_seis_acordaos_entram_no_indice_com_a_propria_chave(self):
        esperados = {
            ("TSE", "AGRAVO REGIMENTAL NO RECURSO ESPECIAL ELEITORAL", "0000374-4220126160066"): 24739990,
            ("TSE", "RECURSO ESPECIAL ELEITORAL", "0000184-7020126200033"): 134380702,
            ("TSE", "RECURSO ESPECIAL ELEITORAL", "0000310-0320126090074"): 24269534,
            ("TST", "AIRR", "0000505-3220105090594"): 1868127437,
            ("TST", "AIRR", "0088600-1320085240005"): 2580912355,
            ("STM", "APELAÇÃO", "0000234-7320167110211"): 2381771667,
        }
        for (tribunal, classe, numero), id_esperado in esperados.items():
            with self.subTest(numero=numero):
                registros = self.indice.get((tribunal, classe, numero, None), [])
                self.assertEqual([r["id"] for r in registros], [id_esperado])

    def test_tst_deixa_de_ser_indexado_pelo_precedente_citado(self):
        # O parser antigo pegava o primeiro "nº TST-..." do texto, que neste
        # documento é um incidente citado, e não o processo do acórdão.
        self.assertNotIn(("TST", "INCJULGRREMBREP", "0001000-7120125060018", None), self.indice)
        registros = self.indice[("TST", "AG-ED-AIRR", "0155000-3620095010030", None)]
        self.assertEqual([r["id"] for r in registros], [2322454299])

    def test_restam_somente_os_dois_tse_sem_cnj_legivel(self):
        self.assertEqual({r["id"] for r in self.sem_parser}, {469075172, 568736504})


if __name__ == "__main__":
    unittest.main()
