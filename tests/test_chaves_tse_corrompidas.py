"""A3: recuperação conservadora de CNJs corrompidos no cabeçalho do TSE."""

import sqlite3
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

import indice_utils as ind
import eda_utils as u
from steps.buscar_na_base import Resolvedor


CASOS_BASE = {
    "0003112-8520146070000": {1888100977},
    "0000487-2620126060049": {23127447},
    "0000598-4920126080018": {576309534},
    "0002622-4720106270000": {433421220},
    "0000102-8120126200019": {23127111, 23531614},
}


class RegraDeRecuperacao(unittest.TestCase):
    def test_cauda_sem_separadores_e_confirmada_pela_repeticao_valida(self):
        texto = "Nº 3112-8520146070000. Mais adiante: 3112-85.2014.6.07.0000."
        self.assertEqual(
            ind._recuperar_cnj_tse("3112-8520146070000", texto),
            "0003112-8520146070000",
        )

    def test_dv_ausente_so_e_recuperado_com_mesmo_sequencial_unico(self):
        texto = "Nº 598- 2012 6 08 0018. Processo correto: 598-49.2012.6.08.0018."
        self.assertEqual(
            ind._recuperar_cnj_tse("598- 2012 6 08 0018", texto),
            "0000598-4920126080018",
        )

    def test_nao_adivinha_sem_confirmacao_ou_com_duas_identidades(self):
        for bruto, texto in (
            ("3112-8520146070000", "Nº 3112-8520146070000, sem repetição íntegra."),
            (
                "598- 2012 6 08 0018",
                "Nº 598- 2012 6 08 0018. 598-49.2012.6.08.0018 e 598-50.2012.6.08.0018.",
            ),
            (
                "3112-8520146070000",
                "A cauda coincide, mas o sequencial é outro: 12-85.2014.6.07.0000.",
            ),
        ):
            with self.subTest(texto=texto):
                self.assertIsNone(ind._recuperar_cnj_tse(bruto, texto))

    def test_numero_comum_e_cnj_ja_valido_nao_entram_na_recuperacao(self):
        self.assertIsNone(ind._recuperar_cnj_tse("1741784", "REsp 1.741.784"))
        self.assertIsNone(
            ind._recuperar_cnj_tse(
                "3112-85.2014.6.07.0000", "3112-85.2014.6.07.0000"
            )
        )


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

    def test_cinco_cnjs_corrigidos_entram_no_indice_por_numero(self):
        for numero, ids_esperados in CASOS_BASE.items():
            with self.subTest(numero=numero):
                ids = {
                    registro["id"]
                    for registro in self.resolvedor.acordaos_por_numero.get(numero, [])
                }
                self.assertEqual(ids, ids_esperados)

    def test_lookup_preserva_unicos_e_a_duplicata(self):
        for numero, ids_esperados in CASOS_BASE.items():
            registros = self.resolvedor.acordaos_por_numero[numero]
            citacao = (
                f"{registros[0]['classe']} nº {u.formatar_cnj_para_fts(numero)} do TSE"
            )
            resultado = self.resolvedor.acordao(citacao)
            with self.subTest(numero=numero):
                if len(ids_esperados) == 1:
                    self.assertEqual(resultado[:2], ("real", next(iter(ids_esperados))))
                else:
                    self.assertEqual(resultado[:2], ("incompleta", None))


if __name__ == "__main__":
    unittest.main()
