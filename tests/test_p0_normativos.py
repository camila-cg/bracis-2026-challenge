"""Regressões P0 contra falsos positivos em citações normativas."""

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor
from steps.normalizar_citacao import normalizar_citacao


def normalizar(trecho: str, familia: str = "lei"):
    return normalizar_citacao(
        {"trecho": trecho, "inicio": 0, "fim": len(trecho)},
        familia,
    )


class IdentidadeLegalEstruturada(unittest.TestCase):
    def setUp(self):
        self.resolvedor = Resolvedor.__new__(Resolvedor)
        self.resolvedor.normativos = {
            "lei|Código de Processo Civil|373": [{"id": 10}],
            "lei|Código de Processo Penal|312": [{"id": 11}],
            "lei|Código Penal Militar|290": [{"id": 12}],
            "lei|CLT|818": [{"id": 13}],
            "sumula|STJ|83": [{"id": 14}],
        }

    def resolver(self, trecho: str, familia: str = "lei"):
        return self.resolvedor.resolver_citacao(normalizar(trecho, familia))

    def test_numero_canonico_aceita_pontuacao_espaco_e_ocr(self):
        casos = (
            "art. 373 da Lei 13.105/2015",
            "art. 373 da Lei 13105/2015",
            "art. 373 da Lei 13 105/2015",
            "art. 373 da Lei I3.lO5/2O15",
            "art. 373 da Lei 13.105, de 16 de março de 2015",
            "art. 373 da Lei 13105",
        )
        for trecho in casos:
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertEqual(
                    (citacao.numero_lei, citacao.diploma, citacao.identidade_legal_valida),
                    ("13105", "Código de Processo Civil", True),
                )
                self.assertEqual(
                    (self.resolver(trecho).classe, self.resolver(trecho).id_canonico),
                    ("real", 10),
                )

    def test_notacao_n_ponto_ordinal_preserva_identidade_legal(self):
        casos = (
            (
                "art. 373 da Lei n.º 13.105/2015",
                "13105",
                "Código de Processo Civil",
            ),
            (
                "art. 1 da Lei Complementar n.º 64/1990",
                "64",
                "Lei Complementar nº 64/1990",
            ),
        )
        for trecho, numero, diploma in casos:
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertEqual(
                    (
                        citacao.numero_lei,
                        citacao.diploma,
                        citacao.identidade_legal_valida,
                    ),
                    (numero, diploma, True),
                )

    def test_ano_numero_extra_e_identidade_contraditoria_nao_resolvem(self):
        casos = (
            "art. 373 da Lei 13.105/2016",
            "art. 373 da Lei 13.1050/2015",
            "art. 373 da Lei 13.105/20150",
            "art. 373 do CPC, segundo a Lei 13.105/2016",
        )
        for trecho in casos:
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertFalse(citacao.identidade_legal_valida)
                self.assertEqual(self.resolver(trecho).classe, "inventada")

    def test_lei_com_qualificador_de_ente_e_numerada_e_sem_mapeamento_logo_inventada(self):
        # Um qualificador de ente entre `Lei` e o número não faz a citação
        # perder o identificador: número explícito sem mapeamento é inventada.
        casos = (
            ("art. 41 da Lei municipal nº 3.918/2016", "3918", "2016"),
            ("art. 18 da Lei federal nº 15.117/2025", "15117", "2025"),
            ("Lei Complementar estadual nº 812/2021, art. 6º, § 4º", "812", "2021"),
            ("9.3 remete ao art. 27 da Lei Nacional nº 14.927/2024", "14927", "2024"),
            ("art. 8º da Lei distrital nº 6.789/2021", "6789", "2021"),
        )
        for trecho, numero, ano in casos:
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertTrue(citacao.lei_numerada)
                self.assertEqual((citacao.numero_lei, citacao.ano_lei), (numero, ano))
                self.assertFalse(citacao.identidade_legal_valida)
                resultado = self.resolver(trecho)
                self.assertEqual((resultado.classe, resultado.caminho), ("inventada", "lei_inventada"))
                self.assertEqual(resultado.motivo, "lei_numerada_sem_mapeamento")

    def test_qualificador_abreviado_ocr_e_ano_de_dois_digitos(self):
        citacao = normalizar("Lei Mun. 2.74l/19, art. 8º")
        self.assertTrue(citacao.lei_numerada)
        self.assertEqual((citacao.numero_lei, citacao.ano_lei), ("2741", "2019"))
        self.assertEqual(self.resolver("Lei Mun. 2.74l/19, art. 8º").classe, "inventada")

    def test_qualificador_federal_ou_nacional_preserva_a_identidade_do_catalogo(self):
        for trecho in (
            "art. 373 da Lei federal nº 13.105/2015",
            "art. 373 da Lei Nacional nº 13.105/2015",
            "art. 373 da Lei Fed. 13.105/2015",
        ):
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertEqual(
                    (citacao.numero_lei, citacao.diploma, citacao.identidade_legal_valida),
                    ("13105", "Código de Processo Civil", True),
                )
                resultado = self.resolver(trecho)
                self.assertEqual((resultado.classe, resultado.id_canonico), ("real", 10))

    def test_lei_estadual_ou_municipal_nao_e_a_lei_federal_do_catalogo(self):
        # O catálogo só tem leis federais; uma lei estadual de mesmo número e
        # ano não é o CPC.
        for trecho in (
            "art. 373 da Lei estadual nº 13.105/2015",
            "art. 373 da Lei municipal nº 13.105/2015",
            "art. 373 da Lei Mun. nº 13.105/2015",
            "art. 373 da Lei distrital nº 13.105/2015",
        ):
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertFalse(citacao.identidade_legal_valida)
                self.assertEqual(self.resolver(trecho).classe, "inventada")

    def test_qualificador_de_ente_sem_numero_nao_vira_lei_numerada(self):
        for trecho in ("art. 5 da Lei estadual pertinente", "o artigo da Lei municipal aplicável"):
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertFalse(citacao.lei_numerada)
                self.assertIsNone(citacao.numero_lei)

    def test_alias_mais_especifico_distingue_cpp_de_cpm(self):
        cpp = normalizar("art. 312 do Código de Processo Penal")
        cpm = normalizar("art. 290 do Código de Processo Penal Militar")

        self.assertEqual(cpp.diploma, "Código de Processo Penal")
        self.assertEqual(cpm.diploma, "Código Penal Militar")
        self.assertEqual(self.resolvedor.resolver_citacao(cpp).id_canonico, 11)
        self.assertEqual(self.resolvedor.resolver_citacao(cpm).id_canonico, 12)

    def test_sufixo_do_artigo_e_preservado_e_rejeitado(self):
        citacao = normalizar("art. 290-B do CPM")

        self.assertEqual(citacao.artigo, "290-B")
        self.assertEqual(self.resolvedor.resolver_citacao(citacao).classe, "inventada")

    def test_ocr_no_meio_do_artigo_nao_e_truncado(self):
        casos = (
            ("art. 3l2 do CPP", "312", "real", 11),
            ("art. 3 l 2 do CPP", "312", "real", 11),
            ("art. 8l8 da CLT", "818", "real", 13),
            ("art. 29O-B do CPM", "290-B", "inventada", None),
        )
        for trecho, artigo, classe, id_canonico in casos:
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                resultado = self.resolvedor.resolver_citacao(citacao)

                self.assertEqual(citacao.artigo, artigo)
                self.assertEqual(
                    (resultado.classe, resultado.id_canonico),
                    (classe, id_canonico),
                )

    def test_paragrafo_e_inciso_nao_participam_da_resolucao(self):
        casos = (
            "art. 818, § 999, da CLT",
            "art. 818, inciso XL, da CLT",
        )
        for trecho in casos:
            with self.subTest(trecho=trecho):
                citacao = normalizar(trecho)
                self.assertEqual(citacao.artigo, "818")
                self.assertEqual(self.resolvedor.resolver_citacao(citacao).id_canonico, 13)

    def test_lei_numerada_desconhecida_ou_zero_e_inventada_sem_artigo(self):
        for trecho in ("Lei 12345", "Lei 000000000000000000000000000000000"):
            with self.subTest(trecho=trecho):
                resultado = self.resolver(trecho)
                self.assertEqual((resultado.classe, resultado.caminho), ("inventada", "lei_inventada"))

    def test_lei_canonica_sem_artigo_continua_incompleta(self):
        resultado = self.resolver("Lei 13.105/2015")
        self.assertEqual((resultado.classe, resultado.caminho), ("incompleta", "lei_sem_artigo"))

    def test_sumula_e_tema_zero_explicitos_sao_inventados(self):
        sumula = self.resolver("Súmula 000 do STJ", "sumula")
        tema = self.resolver("Tema 0", "tema")

        self.assertEqual((sumula.classe, sumula.motivo), ("inventada", "numero_zero"))
        self.assertEqual((tema.classe, tema.motivo), ("inventada", "numero_zero"))

    def test_zero_a_esquerda_nao_vira_sumula_real(self):
        resultado = self.resolver("Súmula 083 do STJ", "sumula")
        self.assertEqual(resultado.classe, "inventada")

    def test_lei_de_murphy_exige_contexto_juridico_forte(self):
        fraca = normalizar("Lei de Murphy")
        forte = normalizar("nos termos do art. 1º da Lei de Murphy")

        self.assertFalse(fraca.contexto_legal_forte)
        self.assertEqual(self.resolvedor.resolver_citacao(fraca).classe, "incompleta")
        self.assertTrue(forte.contexto_legal_forte)
        self.assertEqual(self.resolvedor.resolver_citacao(forte).classe, "inventada")

    def test_pipeline_preserva_nome_de_lei_cortado_fora_do_span(self):
        banco = Path(__file__).resolve().parents[1] / "data" / "desafio1_bracis.db"
        if not banco.is_file():
            self.skipTest("base canônica ausente")
        with sqlite3.connect(banco.as_uri() + "?immutable=1", uri=True) as con:
            resolvedor = Resolvedor(con)
            for nome in ("Murphy", "Murph", "murphy", "murph"):
                with self.subTest(nome=nome):
                    texto = f"Nos termos do art. 1 da Lei de {nome}, decide-se."
                    documento = processar_documento("sintetico", texto, resolvedor)
                    citacoes = [
                        citacao for citacao in documento["citacoes"]
                        if "art. 1" in citacao["trecho"]
                    ]
                    self.assertTrue(citacoes)
                    self.assertTrue(
                        all(citacao["classificacao"] == "inventada" for citacao in citacoes)
                    )


if __name__ == "__main__":
    unittest.main()
