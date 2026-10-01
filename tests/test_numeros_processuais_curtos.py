"""A15: números processuais de 1–2 dígitos em contexto jurídico forte."""

import sqlite3
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor
from steps.normalizar_citacao import normalizar_citacao


CAUTELAR_87 = "Cautelar Inominada Criminal nº 87/DF"
IDS_CAUTELAR_87 = (2548829153, 2931598140, 2939355373)


def normalizar(trecho: str):
    return normalizar_citacao(
        {"trecho": trecho, "inicio": 0, "fim": len(trecho)},
        "acordao",
        ja_normalizado=True,
    )


class ContextoForte(unittest.TestCase):
    def test_classe_e_marcador_explicito_liberam_numero_curto(self):
        citacao = normalizar(CAUTELAR_87)
        self.assertEqual((citacao.classe, citacao.numero, citacao.uf), ("cautelar", "87", "DF"))

    def test_numero_curto_sem_os_dois_sinais_nao_e_consultado(self):
        for trecho in (
            "Cautelar Inominada Criminal 87/DF",
            "Processo nº 87/DF",
            "Classe nº 37 do TSE",
            "O processo tem 87 páginas.",
        ):
            with self.subTest(trecho=trecho):
                self.assertIsNone(normalizar(trecho).numero)

    def test_iac_libera_numero_curto_sem_marcador(self):
        for trecho in ("IAC 15/STJ", "QO no IAC 15/STJ", "IAC 15 do STJ"):
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertEqual(
                    (citacao.classe, citacao.numero),
                    ("incidentedeassuncaodecompetencia", "15"),
                )

    def test_fallback_curto_nao_recorta_prefixo_de_numero_maior(self):
        self.assertEqual(normalizar("Cautelar Inominada Criminal nº 87.123/DF").numero, "87123")


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

    def test_numero_87_chega_ao_lookup_e_preserva_as_tres_duplicatas(self):
        resultado = self.resolvedor.resolver_citacao(normalizar(CAUTELAR_87))
        self.assertEqual(resultado.classe, "incompleta")
        self.assertEqual(resultado.caminho, "acordao_ambigua")
        self.assertEqual(resultado.candidatos, IDS_CAUTELAR_87)

    def test_pipeline_extrai_a_classe_e_consulta_o_numero_curto(self):
        for forma in (CAUTELAR_87, "Cautelar Inominada Criminal nº 87 do STJ"):
            with self.subTest(forma=forma):
                texto = f"Nesse sentido, confira-se o que decidiu esta Corte na {forma}, aplicável ao caso."
                citacoes = processar_documento("sintetico", texto, self.resolvedor)["citacoes"]
                self.assertEqual(len(citacoes), 1)
                self.assertTrue(citacoes[0]["trecho"].startswith("Cautelar Inominada Criminal"))
                self.assertEqual(
                    (citacoes[0]["classificacao"], citacoes[0]["caminho"]),
                    ("incompleta", "acordao_ambigua"),
                )

    def test_quantidade_curta_em_prosa_nao_abre_citacao(self):
        for texto in (
            "A medida cautelar foi deferida por 87 dias, conforme a prova dos autos.",
            "A Cautelar Inominada Criminal 87/DF foi mencionada sem marcador numérico.",
            "O recurso cautelar menciona a Cautelar Inominada Criminal no 87º parágrafo.",
        ):
            with self.subTest(texto=texto):
                self.assertEqual(
                    processar_documento("sintetico", texto, self.resolvedor)["citacoes"],
                    [],
                )

    def test_numero_curto_inexistente_continua_inventado(self):
        resultado = self.resolvedor.resolver_citacao(
            normalizar("Cautelar Inominada Criminal nº 42/DF")
        )
        self.assertEqual((resultado.classe, resultado.caminho), ("inventada", "acordao_zero"))

    def test_iac_com_numero_curto_sem_marcador_chega_ao_lookup(self):
        # A base não tem IAC: com o número consultado, a resposta é inventada,
        # não incompleta por número ausente.
        for forma in ("QO no IAC 15/STJ", "IAC 15/STJ", "IAC 15 do STJ", "IAC nº 15 do STJ"):
            with self.subTest(forma=forma):
                texto = f"A tese foi firmada no {forma}, aplicável ao caso."
                citacoes = processar_documento("sintetico", texto, self.resolvedor)["citacoes"]
                self.assertEqual(len(citacoes), 1)
                self.assertTrue(citacoes[0]["trecho"].startswith(forma.split()[0]))
                self.assertEqual(
                    (citacoes[0]["classificacao"], citacoes[0]["caminho"]),
                    ("inventada", "acordao_zero"),
                )

    def test_iac_por_extenso_e_detectado_e_consultado(self):
        for forma in (
            "Incidente de Assunção de Competência 15/STJ",
            "Incidente de Assunção de Competência nº 15 do STJ",
            "Incidente de Assunção de Competência n. 15/STJ",
            "Questão de Ordem no Incidente de Assunção de Competência 15/STJ",
        ):
            with self.subTest(forma=forma):
                texto = f"A tese foi firmada no {forma}, aplicável ao caso."
                citacoes = processar_documento("sintetico", texto, self.resolvedor)["citacoes"]
                self.assertEqual([c["trecho"] for c in citacoes], [forma])
                self.assertEqual(
                    (citacoes[0]["classificacao"], citacoes[0]["caminho"]),
                    ("inventada", "acordao_zero"),
                )

    def test_numero_antes_do_iac_nao_abre_citacao(self):
        # "o IAC" sem número próprio é menção vaga; o número da frase não é dele.
        for texto in (
            "Foram 15 casos no IAC.",
            "Há 3 votos no IAC do STJ.",
            "O IAC foi instaurado.",
        ):
            with self.subTest(texto=texto):
                self.assertEqual(
                    processar_documento("sintetico", texto, self.resolvedor)["citacoes"],
                    [],
                )

    def test_iac_com_sufixo_que_nao_e_tribunal_nem_uf_nao_e_consultado(self):
        citacoes = processar_documento(
            "sintetico", "Aplica-se o IAC 15/XX ao caso.", self.resolvedor
        )["citacoes"]
        self.assertNotEqual([c["caminho"] for c in citacoes], ["acordao_zero"])


if __name__ == "__main__":
    unittest.main()
