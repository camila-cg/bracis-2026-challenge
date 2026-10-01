"""Trava as premissas da base usadas para indexar dispositivos e súmulas."""

import re
import sqlite3
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import indice_utils as ind

QUANTIDADE_DISPOSITIVOS = 13
QUANTIDADE_SUMULAS = 5


class BaseCanonicaNormativos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        banco = RAIZ / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        cls.con = sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True)
        cls.registros = cls.con.execute(
            "SELECT documento_id, id, natureza, texto FROM documentos "
            "WHERE natureza IN ('dispositivo', 'sumula')"
        ).fetchall()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "con"):
            cls.con.close()

    def test_quantidades_de_normativos_na_base_nao_mudam_sem_revisao(self):
        por_natureza = {
            natureza: sum(registro[2] == natureza for registro in self.registros)
            for natureza in ("dispositivo", "sumula")
        }
        self.assertEqual(por_natureza["dispositivo"], QUANTIDADE_DISPOSITIVOS)
        self.assertEqual(por_natureza["sumula"], QUANTIDADE_SUMULAS)

    def test_dispositivo_tem_titulo_e_artigo_legiveis(self):
        for documento_id, id_canonico, natureza, texto in self.registros:
            if natureza != "dispositivo":
                continue
            with self.subTest(documento_id=documento_id, id=id_canonico):
                primeira_linha, *demais_linhas = texto.splitlines()
                titulo = re.match(r"\s*Artigo\s+(\d[\d.]*)", primeira_linha, re.I)
                self.assertIsNotNone(titulo, primeira_linha)

                artigo_titulo = titulo.group(1).replace(".", "")
                artigo_indice = ind.ARTIGO_INICIAL.match(texto)
                self.assertIsNotNone(artigo_indice, primeira_linha)
                self.assertEqual(
                    artigo_indice.group(1).replace(".", ""), artigo_titulo
                )

                artigo_no_corpo = re.compile(
                    rf"\s*Art\.\s*{re.escape(titulo.group(1))}(?:[º°])?(?:\D|$)",
                    re.I,
                )
                self.assertTrue(
                    any(artigo_no_corpo.match(linha) for linha in demais_linhas),
                    f"artigo {artigo_titulo} ausente no corpo de {documento_id}",
                )

    def test_indice_resolve_todos_os_normativos_sem_colisao(self):
        indice, sem_resolucao = ind.construir_indice_normativos(self.con)

        self.assertEqual(sem_resolucao, [])
        self.assertEqual(len(indice), QUANTIDADE_DISPOSITIVOS + QUANTIDADE_SUMULAS)
        self.assertEqual(sum(map(len, indice.values())), len(self.registros))
        self.assertTrue(all(len(registros) == 1 for registros in indice.values()))
        self.assertEqual(
            sum(chave.startswith("lei|") for chave in indice),
            QUANTIDADE_DISPOSITIVOS,
        )
        self.assertEqual(
            sum(chave.startswith("sumula|") for chave in indice),
            QUANTIDADE_SUMULAS,
        )


if __name__ == "__main__":
    unittest.main()
