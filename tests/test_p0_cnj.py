"""Regressões P0: um CNJ adulterado nunca pode casar por substring."""

import sqlite3
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import eda_utils as u
import indice_utils as ind
from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor
from steps.normalizar_citacao import normalizar_citacao
from steps.normalizar_numeros import normalizar_trecho
from steps.rotear_citacao import ROTA_ACORDAO


CNJ_REAL = "7000136-79.2023.7.00.0000"
CHAVE_REAL = "7000136-7920237000000"


class NormalizacaoEstritaCnj(unittest.TestCase):
    def test_cnj_valido_e_sequencial_abreviado_continuam_canonicos(self):
        self.assertEqual(u.chave_cnj(CNJ_REAL), CHAVE_REAL)
        self.assertEqual(
            u.variantes_numero("310-76.2016.6.16.0006"),
            ["310-76.2016.6.16.0006", "0000310-76.2016.6.16.0006"],
        )

    def test_chave_canonica_tem_formatacao_segura_para_fts(self):
        self.assertEqual(u.formatar_cnj_para_fts(CHAVE_REAL), CNJ_REAL)
        self.assertEqual(u.variantes_numero(CHAVE_REAL), [CNJ_REAL])

    def test_chave_canonica_preserva_variante_curta_para_fallback_fts(self):
        chave_curta_canonica = "0000310-7620166160006"

        self.assertEqual(
            u.variantes_numero(chave_curta_canonica),
            [
                "0000310-76.2016.6.16.0006",
                "310-76.2016.6.16.0006",
            ],
        )

    def test_sequencial_com_mais_de_sete_digitos_nao_casa_por_substring(self):
        adulterado = "97000136-79.2023.7.00.0000"
        self.assertIsNone(u.chave_cnj(adulterado))
        self.assertIsNone(u.formatar_cnj_para_fts(adulterado))
        self.assertEqual(u.variantes_numero(adulterado), [])
        self.assertNotEqual(ind.normalizar_numero(adulterado), CHAVE_REAL)

    def test_digito_extra_adjacente_ao_final_invalida_o_bloco_inteiro(self):
        adulterado = CNJ_REAL + "9"
        self.assertIsNone(u.chave_cnj(adulterado))
        self.assertEqual(u.variantes_numero(adulterado), [])
        self.assertNotEqual(ind.normalizar_numero(adulterado), CHAVE_REAL)

    def test_digito_extra_separado_por_pontuacao_nao_casa_por_substring(self):
        for adulterado in (
            "9." + CNJ_REAL,
            "9-" + CNJ_REAL,
            "9/" + CNJ_REAL,
            CNJ_REAL + ".9",
            CNJ_REAL + "-9",
            CNJ_REAL + "/9",
            "9.-" + CNJ_REAL,
            "9–" + CNJ_REAL,
            "9—" + CNJ_REAL,
            CNJ_REAL + "--9",
            CNJ_REAL + "–9",
            CNJ_REAL + "—9",
        ):
            with self.subTest(adulterado=adulterado):
                self.assertEqual(u.variantes_numero(adulterado), [])

    def test_alteracao_interna_produz_outra_identidade(self):
        adulterado = "7000136-79.2024.7.00.0000"
        self.assertEqual(u.chave_cnj(adulterado), "7000136-7920247000000")
        self.assertNotEqual(u.chave_cnj(adulterado), CHAVE_REAL)

    def test_ocr_reversivel_permanece_suportado(self):
        com_ocr = "7OOO136-79.2O23.7.OO.OOOO"
        self.assertEqual(u.chave_cnj(com_ocr), CHAVE_REAL)
        self.assertEqual(u.variantes_numero(com_ocr), [CNJ_REAL])
        self.assertEqual(normalizar_trecho(com_ocr), CNJ_REAL)

    def test_normalizacao_nao_formata_sequencial_de_oito_digitos_como_cnj(self):
        normalizado = normalizar_trecho("APL 97000136-79.2023.7.00.0000/RS")
        self.assertIsNone(u.CNJ.search(normalizado))
        self.assertEqual(u.variantes_numero(normalizado), [])

    def test_cnj_adulterado_e_reconhecido_pela_estrutura(self):
        for adulterado in (
            "97000136-79.2023.7.00.0000",
            "70001360-79.2023.7.00.0000",
            CNJ_REAL + ".1",
            CNJ_REAL + "9",
            "9." + CNJ_REAL,
            CNJ_REAL + "/9",
        ):
            with self.subTest(adulterado=adulterado):
                self.assertTrue(u.cnj_adulterado(f"APL {adulterado}/RS"))

    def test_cnj_valido_ou_sem_estrutura_de_cnj_nao_e_adulterado(self):
        for trecho in (
            f"APL {CNJ_REAL}/RS",
            "310-76.2016.6.16.0006",
            "7OOO136-79.2O23.7.OO.OOOO",
            "REsp 1.741.784/PR do STJ",
            "REsp 1.741.784/PR, fls. 12-14.",
            "Reclamação nº 66.516/RO",
            "julgado do STF proferido em 2024 pela relatoria de Dias Toffoli",
        ):
            with self.subTest(trecho=trecho):
                self.assertFalse(u.cnj_adulterado(trecho))


class ResolucaoEstritaCnj(unittest.TestCase):
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

    def test_cnj_real_e_abreviado_continuam_resolvendo(self):
        self.assertEqual(
            self.resolvedor.acordao(f"APL {CNJ_REAL}/RS")[:2],
            ("real", 2161632985),
        )
        self.assertEqual(
            self.resolvedor.acordao("310-76.2016.6.16.0006")[:2],
            ("real", 439317631),
        )

    def test_prefixo_digito_extra_nao_resolve_para_o_cnj_real(self):
        classe, id_canonico, _ = self.resolvedor.acordao(
            "APL 97000136-79.2023.7.00.0000/RS"
        )
        self.assertNotEqual(classe, "real")
        self.assertIsNone(id_canonico)

    def test_alteracao_interna_nao_resolve_para_o_cnj_real(self):
        self.assertEqual(
            self.resolvedor.acordao("APL 7000136-79.2024.7.00.0000/RS")[:2],
            ("inventada", None),
        )

    def test_cnj_malformado_e_identificador_explicito_invalido_logo_inventada(self):
        # Identificador explícito inválido não é identificador ausente: não pode
        # cair na rota descritiva (incompleta).
        for adulterado in (
            "APL 97000136-79.2023.7.00.0000/RS",
            "APL 17000136-79.2023.7.00.0000/RS",
            "APL 70001360-79.2023.7.00.0000/RS",
            f"APL {CNJ_REAL}.1/RS",
            f"APL {CNJ_REAL}9/RS",
            f"APL 9.{CNJ_REAL}/RS",
            f"APL {CNJ_REAL}-9/RS",
        ):
            with self.subTest(adulterado=adulterado):
                classe, id_canonico, caminho = self.resolvedor.acordao(adulterado)
                self.assertEqual(classe, "inventada")
                self.assertIsNone(id_canonico)
                self.assertEqual(caminho, "acordao_zero")

    def test_cnj_malformado_informa_o_motivo(self):
        citacao = normalizar_citacao(
            {"trecho": "APL 97000136-79.2023.7.00.0000/RS", "inicio": 0, "fim": 34},
            ROTA_ACORDAO,
            ja_normalizado=True,
        )
        self.assertTrue(citacao.identificador_adulterado)
        resultado = self.resolvedor.resolver_citacao(citacao)
        self.assertEqual(resultado.classe, "inventada")
        self.assertEqual(resultado.motivo, "cnj_malformado")
        self.assertEqual(resultado.cobertura, "fora_base")

    def test_cnj_valido_e_numero_comum_nao_sao_marcados_como_adulterados(self):
        for trecho in (f"APL {CNJ_REAL}/RS", "REsp 1.741.784/PR do STJ", "Reclamação nº 66.516/RO"):
            with self.subTest(trecho=trecho):
                citacao = normalizar_citacao(
                    {"trecho": trecho, "inicio": 0, "fim": len(trecho)},
                    ROTA_ACORDAO,
                    ja_normalizado=True,
                )
                self.assertFalse(citacao.identificador_adulterado)

    def test_documento_ponta_a_ponta_classifica_cnj_malformado_como_inventada(self):
        texto = "Conforme a APL 97000136-79.2023.7.00.0000/RS, a parte deve recorrer."
        citacoes = processar_documento("sintetico", texto, self.resolvedor)["citacoes"]
        self.assertEqual(len(citacoes), 1)
        self.assertEqual(citacoes[0]["classificacao"], "inventada")
        self.assertEqual(citacoes[0]["caminho"], "acordao_zero")


if __name__ == "__main__":
    unittest.main()
