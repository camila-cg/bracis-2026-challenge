"""Gate do teste da base inteira (src/validacao/teste_base_inteira.py)."""

import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import eda_utils as u
from validacao import teste_base_inteira as t

# Medidos em 01/10/2026 com a frase de citação realista (atualizados após o A10). Regrediu abaixo dos
# pisos (ou acima dos tetos): a mudança piorou a generalização. Melhorou: mova o
# limite na mesma mudança.
PISO_OK_POR_TRIBUNAL = {"STF": 198, "STJ": 174, "STM": 172, "TSE": 118, "TST": 176}
TETO_RESPOSTA_ERRADA = 0          # real_errado + inventada_errado
TETO_NAO_EXTRAIU = 2
TETO_EXTRAS_NA_FRASE = 0
TETO_PREVISOES_EM_DISTRATORES = 0


class FraseDeCitacao(unittest.TestCase):
    def test_classe_do_tse_perde_o_ruido_do_cabecalho(self):
        for bruta, esperada in (
            ("TRIBUNAL SUPERIOR ELEITORAL ACÓRDÃO RECURSO ESPECIAL ELEITORAL", "Recurso Especial Eleitoral"),
            ("PUBLICADO EM SESSÃO TRIBUNAL SUPERIOR ELEITORAL ACÓRDÃO AGRAVO REGIMENTAL NO RECURSO ESPECIAL ELEITORAL",
             "Agravo Regimental No Recurso Especial Eleitoral"),
            ("SUPERIOR ELEITORAL ACÓRDÃO RECURSO ESPECIAL ELEITORAL", "Recurso Especial Eleitoral"),
            ("PRIMEIRA TURMA AG.REG. NO RE", "Ag.Reg. No Re"),
        ):
            with self.subTest(bruta=bruta):
                self.assertEqual(t.classe_para_citacao(bruta, "TSE"), esperada)

    def test_sigla_de_classe_do_tst_nao_vira_title_case(self):
        self.assertEqual(t.classe_para_citacao("AIRR", "TST"), "AIRR")
        self.assertEqual(t.classe_para_citacao("Ag-AIRR", "TST"), "Ag-AIRR")

    def test_cnj_entra_pontuado_e_numero_comum_agrupado_por_milhares(self):
        self.assertEqual(t.numero_para_citacao("0600393-6720206060075"), "0600393-67.2020.6.06.0075")
        self.assertEqual(t.numero_para_citacao("1741784"), "1.741.784")

    def test_uf_so_aparece_quando_o_cabecalho_a_tem(self):
        com_uf = t.frase_de_citacao("Recurso Especial", "1741784", "SP", "STJ")
        sem_uf = t.frase_de_citacao("RR", "0001234-5620175010001", None, "TST")
        self.assertIn("1.741.784/SP", com_uf)
        self.assertNotIn("/DF", sem_uf)
        self.assertNotIn("/None", sem_uf)

    def test_numero_corrompido_so_com_digitos_e_comprimento_de_cnj(self):
        for corrompido in ("31128520146070000", "4872620126060049", "59820126080018"):
            with self.subTest(numero=corrompido):
                self.assertTrue(t.numero_corrompido(corrompido))
        for normal in ("1741784", "0600393-6720206060075", "2668"):
            with self.subTest(numero=normal):
                self.assertFalse(t.numero_corrompido(normal))


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (RAIZ / "data" / "desafio1_bracis.db").is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        cls.medicao = t.executar(u.conectar_base())

    def test_acertos_por_tribunal_nao_regridem(self):
        acertos = t.ok_por_tribunal(self.medicao)
        for tribunal, piso in PISO_OK_POR_TRIBUNAL.items():
            with self.subTest(tribunal=tribunal):
                self.assertGreaterEqual(acertos[tribunal][0], piso)

    def test_respostas_erradas_e_nao_extraidas_nao_aumentam(self):
        resultado = self.medicao["resultado"]
        erradas = resultado["real_errado"] + resultado["inventada_errado"]
        self.assertLessEqual(erradas, TETO_RESPOSTA_ERRADA)
        self.assertLessEqual(resultado["nao_extraiu"], TETO_NAO_EXTRAIU)

    def test_falsos_positivos_nao_aumentam(self):
        self.assertLessEqual(self.medicao["extras"], TETO_EXTRAS_NA_FRASE)
        self.assertLessEqual(self.medicao["fp_distratores"], TETO_PREVISOES_EM_DISTRATORES)

if __name__ == "__main__":
    unittest.main()
