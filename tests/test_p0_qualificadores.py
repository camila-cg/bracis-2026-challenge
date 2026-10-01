"""Regressões P0 para identificadores e qualificadores de citações.

Os testes deste módulo exercitam a base canônica real para garantir que ruídos
de contexto e qualificadores explícitos nunca transformem uma citação
contraditória em ``real``.
"""

import sqlite3
import unittest
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]

import eda_utils as u
from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor
from steps.normalizar_citacao import normalizar_citacao


def _normalizar_acordao(trecho: str):
    """Normaliza um trecho de acórdão pelo mesmo fluxo usado no pipeline."""
    return normalizar_citacao(
        {"inicio": 0, "fim": len(trecho), "trecho": trecho},
        "acordao",
    )


def _normalizar_sumula(trecho: str):
    """Normaliza um trecho de súmula preservando sua superfície original."""
    return normalizar_citacao(
        {"inicio": 0, "fim": len(trecho), "trecho": trecho},
        "sumula",
    )


class QualificadoresP0(unittest.TestCase):
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

    def test_ano_contextual_antes_do_resp_nao_substitui_numero_do_processo(self):
        trecho = "Precedente de 2023: REsp 1.741.784/PR"

        citacao = _normalizar_acordao(trecho)
        resultado = self.resolvedor.resolver_citacao(citacao)

        self.assertEqual(citacao.numero, "1741784")
        self.assertEqual((resultado.classe, resultado.id_canonico), ("real", 2566535283))

    def test_ano_contextual_entre_classe_e_marcador_nao_vira_processo(self):
        for trecho in (
            "REsp julgado em 2023, 1.741.784/PR do STJ",
            "REsp julgado em 12/03/2023, 1.741.784/PR do STJ",
            "REsp julgado em 2023, nº 1.741.784/PR do STJ",
            "REsp, de 2023, n. 1.741.784/PR do STJ",
            "REsp 2023 1.741.784/PR do STJ",
        ):
            with self.subTest(trecho=trecho):
                citacao = _normalizar_acordao(trecho)
                resultado = self.resolvedor.resolver_citacao(citacao)

                self.assertEqual(citacao.numero, "1741784")
                self.assertEqual(
                    (resultado.classe, resultado.id_canonico),
                    ("real", 2566535283),
                )

    def test_tribunal_explicito_desconhecido_nao_e_tratado_como_ausente(self):
        for tribunal in (
            "STK", "ST-K", "TJSP", "TJ-SP", "TJ SP", "T.J.S.P.", "TRF-99",
        ):
            with self.subTest(tribunal=tribunal):
                resultado = self.resolvedor.acordao(
                    f"REsp 1.741.784/PR do {tribunal}"
                )

                self.assertEqual(resultado[:2], ("inventada", None))

    def test_tribunal_conhecido_pontuado_e_normalizado(self):
        citacao = _normalizar_acordao("REsp 1.741.784/PR do S.T.J.")
        resultado = self.resolvedor.resolver_citacao(citacao)

        self.assertEqual(citacao.tribunal, "STJ")
        self.assertEqual(
            (resultado.classe, resultado.id_canonico), ("real", 2566535283)
        )

    def test_tribunal_conhecido_divergente_continua_inventado(self):
        resultado = self.resolvedor.acordao("REsp 1.741.784/PR do STF")

        self.assertEqual(resultado[:2], ("inventada", None))

    def test_uf_conhecida_sem_espacos_continua_sendo_filtro(self):
        citacao = _normalizar_acordao("REsp 1.741.784-SC do STJ")
        resultado = self.resolvedor.resolver_citacao(citacao)

        self.assertEqual(citacao.uf, "SC")
        self.assertEqual((resultado.classe, resultado.id_canonico), ("inventada", None))

    def test_uf_explicita_desconhecida_nao_e_tratada_como_ausente(self):
        for trecho in (
            "REsp 1.741.784/XX do STJ",
            "310-76.2016.6.16.0006 - XX",
            "310-76.2016.6.16.0006 (XX)",
        ):
            with self.subTest(trecho=trecho):
                citacao = _normalizar_acordao(trecho)
                resultado = self.resolvedor.resolver_citacao(citacao)

                self.assertEqual(citacao.qualificador_invalido, "XX")
                self.assertEqual(
                    (resultado.classe, resultado.id_canonico),
                    ("inventada", None),
                )

    def test_qualificadores_conflitantes_nao_resolvem_como_reais(self):
        for trecho in (
            "REsp 1.741.784/PR do STJ e STF",
            "REsp 1.741.784/PR e /SC do STJ",
        ):
            with self.subTest(trecho=trecho):
                resultado = self.resolvedor.acordao(trecho)
                self.assertEqual(resultado[:2], ("inventada", None))

    def test_pipeline_considera_tribunal_desconhecido_fora_do_span(self):
        textos = (
            "Conforme REsp 1.741.784/PR do STK, o entendimento prevalece.",
            "Conforme REsp 1.741.784/PR do TJSP, o entendimento prevalece.",
            "Conforme REsp 1.741.784/PR do TJ-SP, o entendimento prevalece.",
            "STK, REsp 1.741.784/PR confirma o entendimento.",
        )
        for texto in textos:
            with self.subTest(texto=texto):
                documento = processar_documento("sintetico", texto, self.resolvedor)
                citacoes = [
                    citacao for citacao in documento["citacoes"]
                    if "1.741.784" in citacao["trecho"]
                ]

                self.assertTrue(citacoes)
                self.assertTrue(
                    all(citacao["classificacao"] == "inventada" for citacao in citacoes)
                )

    def test_sumula_seguida_de_tribunal_nao_perde_campos_por_correcao_ocr(self):
        for trecho in ("Súmula 83 STJ", "Súmula 83.STJ"):
            with self.subTest(trecho=trecho):
                citacao = _normalizar_sumula(trecho)
                resultado = self.resolvedor.resolver_citacao(citacao)

                self.assertEqual((citacao.numero, citacao.tribunal), ("83", "STJ"))
                self.assertEqual(
                    (resultado.classe, resultado.id_canonico),
                    ("real", 1289710642),
                )

    def test_resp_seguido_de_tribunal_nao_perde_campos_por_correcao_ocr(self):
        for trecho in ("REsp 1.741.784 STJ", "REsp 1.741.784.STJ"):
            with self.subTest(trecho=trecho):
                citacao = _normalizar_acordao(trecho)
                resultado = self.resolvedor.resolver_citacao(citacao)

                self.assertEqual(
                    (citacao.numero, citacao.tribunal), ("1741784", "STJ")
                )
                self.assertEqual(
                    (resultado.classe, resultado.id_canonico),
                    ("real", 2566535283),
                )

    def test_correcao_ocr_nao_consome_prefixos_processuais(self):
        for trecho in ("TST-123", "TST-123.456", "TST123", "S-83"):
            with self.subTest(trecho=trecho):
                self.assertEqual(u.corrigir_ocr_em_numeros(trecho), trecho)

    def test_correcao_ocr_aceita_sequencias_internas(self):
        casos = {
            "1OO": "100",
            "10OO": "1000",
            "1.OOO.123": "1.000.123",
            "1 OOO 123": "1 000 123",
            "1. OOO. 123": "1. 000. 123",
            "1\nOOO\n123": "1\n000\n123",
        }
        for trecho, esperado in casos.items():
            with self.subTest(trecho=trecho):
                self.assertEqual(u.corrigir_ocr_em_numeros(trecho), esperado)

        citacao = _normalizar_acordao("REsp 1.OOO.123/PR")
        self.assertEqual(citacao.numero, "1000123")
        citacao_com_espacos = _normalizar_acordao("REsp 1 OOO 123/PR")
        self.assertEqual(citacao_com_espacos.numero, "1000123")

    def test_modificadores_processuais_nao_sao_ufs_invalidas(self):
        for trecho in (
            "RE 123-AgR",
            "RE 123-ED",
            "RE 123-QO",
            "AREsp 1.234-AgInt",
        ):
            with self.subTest(trecho=trecho):
                citacao = _normalizar_acordao(trecho)
                self.assertIsNone(citacao.qualificador_invalido)

        citacao_real = _normalizar_acordao("REsp 1.741.784-AgInt do STJ")
        resultado = self.resolvedor.resolver_citacao(citacao_real)
        self.assertEqual((resultado.classe, resultado.id_canonico), ("real", 2566535283))

    def test_qualificadores_ausentes_ainda_permitem_resolver_acordao_unico(self):
        casos = (
            "REsp 1.741.784/PR",       # tribunal ausente
            "REsp 1.741.784 do STJ",  # UF ausente
            "REsp 1.741.784",         # tribunal e UF ausentes
        )

        for trecho in casos:
            with self.subTest(trecho=trecho):
                self.assertEqual(
                    self.resolvedor.acordao(trecho)[:2],
                    ("real", 2566535283),
                )

    def test_numeros_extremos_nao_quebram_nem_resolvem_como_reais(self):
        numero_extremo = "0" * 256
        casos = (
            ("acordao", f"REsp {numero_extremo}/PR do STJ"),
            ("sumula", f"Súmula {numero_extremo} do STJ"),
        )

        for familia, trecho in casos:
            with self.subTest(familia=familia):
                citacao = normalizar_citacao(
                    {"inicio": 0, "fim": len(trecho), "trecho": trecho},
                    familia,
                    ja_normalizado=familia == "acordao",
                )
                resultado = self.resolvedor.resolver_citacao(citacao)

                self.assertIn(resultado.classe, {"inventada", "incompleta"})
                self.assertIsNone(resultado.id_canonico)


if __name__ == "__main__":
    unittest.main()
