"""Regressões P0: um CNJ adulterado nunca pode casar por substring."""

import sqlite3
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

import eda_utils as u
import indice_utils as ind
from pipeline import processar_documento
from steps.buscar_na_base import Resolvedor
from steps.normalizar_citacao import _remover_anos_contextuais, normalizar_citacao
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

    def test_tracos_tipograficos_entre_digitos_valem_como_hifen_no_cnj(self):
        # OCR e editores trocam o hífen do CNJ por travessão, menos etc. O número
        # é o mesmo; só o glifo mudou.
        for glifo in ("‐", "‑", "‒", "–", "—", "―", "−"):
            with self.subTest(glifo=f"U+{ord(glifo):04X}"):
                com_glifo = f"7000136{glifo}79.2023.7.00.0000"
                self.assertEqual(u.chave_cnj(com_glifo), CHAVE_REAL)
                self.assertEqual(u.variantes_numero(com_glifo), [CNJ_REAL])
                self.assertFalse(u.cnj_adulterado(f"APL {com_glifo}/RS"))

    def test_travessao_espacado_tambem_vale_como_hifen_no_cnj(self):
        self.assertEqual(u.variantes_numero("7000136 — 79.2023.7.00.0000"), [CNJ_REAL])
        self.assertEqual(
            u.variantes_numero("310 — 76.2016.6.16.0006"),
            ["310-76.2016.6.16.0006", "0000310-76.2016.6.16.0006"],
        )

    def test_cnj_partido_nao_entrega_o_sequencial_como_numero_livre(self):
        # Se a cauda DV.ano.J.TR.origem existe mas o CNJ não fecha, sobrar só o
        # sequencial (``1036``) seria mandar um bloco curto ao FTS.
        for partido in (
            "1036~78.2013.5.03.0110",
            "1036/78.2013.5.03.0110",
            "1036 ~ 78.2013.5.03.0110",
        ):
            with self.subTest(partido=partido):
                self.assertEqual(u.variantes_numero(partido), [])
                self.assertTrue(u.cnj_adulterado(f"TST RR {partido}"))

    def test_traco_so_vira_hifen_quando_a_cauda_do_cnj_vem_depois(self):
        # Intervalos e datas seguem como estavam, e a correção colada não muda o
        # comprimento (há quem use posições do resultado).
        for texto in ("fls. 12—15", "REsp 1.741.784/PR, pág. 12—15.", "12—05—2023"):
            with self.subTest(texto=texto):
                self.assertEqual(u.corrigir_ocr_em_numeros(texto), texto)
                self.assertEqual(u.normalizar_traco_de_cnj(texto), texto)
        self.assertFalse(u.cnj_adulterado("REsp 1.741.784/PR, pág. 12—15."))
        colado = f"APL 7000136—79.2023.7.00.0000/RS"
        self.assertEqual(len(u.corrigir_ocr_em_numeros(colado)), len(colado))
        self.assertEqual(u.corrigir_ocr_em_numeros(colado), f"APL {CNJ_REAL}/RS")

    def test_normalizacao_do_trecho_preserva_a_cauda_de_cnj_partido(self):
        for partido in ("TST RR 1036~78.2013.5.03.0110", "TST RR 1036/78.2013.5.03.0110"):
            with self.subTest(partido=partido):
                self.assertIn("78.2013.5.03.0110", normalizar_trecho(partido))
        self.assertEqual(
            normalizar_trecho("TST RR 1036 — 78.2013.5.03.0110"),
            "TST RR 1036-78.2013.5.03.0110",
        )

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


# CNJs do cenário de ruído do estresse: letra de OCR no lugar de um zero, quebra
# de linha no meio e, no segundo, um ponto perdido.
CNJ_COM_OCR_E_QUEBRA = "RSE nº 7000768-37.\n2025.7.0O.0000 - RS"
CNJ_COM_OCR_E_PONTO_PERDIDO = "APL 7000136-79.\n2023.7.00O000 - RS"


class AnosContextuaisNaoApagamCnj(unittest.TestCase):
    """`ano.segmento.tribunal` de um CNJ com letra de OCR não é uma data."""

    def test_cnj_com_letra_de_ocr_mantem_o_ano_e_o_comprimento(self):
        for trecho in (CNJ_COM_OCR_E_QUEBRA, CNJ_COM_OCR_E_PONTO_PERDIDO,
                       "RSE nº 7000768-37.2025.7.0O.0000", "APL 7000136-79.2023.7.OO.0000"):
            with self.subTest(trecho=trecho):
                resultado = _remover_anos_contextuais(trecho)
                self.assertEqual(len(resultado), len(trecho))
                self.assertRegex(resultado, r"20(?:25|23)\.7")

    def test_anos_e_datas_de_contexto_continuam_sendo_apagados(self):
        # Caracterização: a correção não pode reabrir o erro que a função resolve.
        casos = (
            ("REsp 2023 1.741.784/PR", "2023"),
            ("Precedente de 2023: REsp 1.741.784/PR do STJ", "2023"),
            ("julgado em 12/05/2023, REsp 1.741.784/PR", "2023"),
            ("julgado em 2023.5.12, REsp 1.741.784/PR", "2023"),
        )
        for trecho, ano in casos:
            with self.subTest(trecho=trecho):
                resultado = _remover_anos_contextuais(trecho)
                self.assertNotIn(ano, resultado)
                self.assertIn("1.741.784", resultado)
                self.assertEqual(len(resultado), len(trecho))

    def test_cnj_valido_ou_sem_ruido_nao_muda(self):
        for trecho in (f"APL {CNJ_REAL}/RS", "310-76.2016.6.16.0006", "REsp 1.741.784/PR do STJ"):
            with self.subTest(trecho=trecho):
                self.assertEqual(_remover_anos_contextuais(trecho), trecho)


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

    def test_cnj_com_traco_tipografico_resolve_para_o_mesmo_id_que_com_hifen(self):
        for glifo in ("‐", "‑", "‒", "–", "—", "―", "−"):
            with self.subTest(glifo=f"U+{ord(glifo):04X}"):
                self.assertEqual(
                    self.resolvedor.acordao(f"TST RR 1036{glifo}78.2013.5.03.0110, 5ª T., rel. Min. Douglas Alencar Rodrigues"),
                    ("real", 2663361645, "acordao_indice"),
                )
                self.assertEqual(
                    self.resolvedor.acordao(f"APL 7000136{glifo}79.2023.7.00.0000/RS")[:2],
                    ("real", 2161632985),
                )

    def test_cnj_adulterado_com_traco_tipografico_continua_inventada(self):
        for glifo in ("–", "—"):
            with self.subTest(glifo=f"U+{ord(glifo):04X}"):
                classe, id_canonico, caminho = self.resolvedor.acordao(
                    f"APL 97000136{glifo}79.2023.7.00.0000/RS"
                )
                self.assertEqual((classe, id_canonico, caminho), ("inventada", None, "acordao_zero"))

    def test_cnj_partido_nunca_resolve_por_um_bloco_curto(self):
        for partido in ("1036~78.2013.5.03.0110", "1036/78.2013.5.03.0110"):
            with self.subTest(partido=partido):
                classe, id_canonico, caminho = self.resolvedor.acordao(f"TST RR {partido}, 5ª T.")
                self.assertEqual((classe, id_canonico, caminho), ("inventada", None, "acordao_zero"))

    def test_documento_ponta_a_ponta_com_travessao_resolve_o_acordao_certo(self):
        texto = "Conforme o TST RR 1036—78.2013.5.03.0110, 5ª T., rel. Min. Douglas Alencar Rodrigues."
        citacoes = processar_documento("sintetico", texto, self.resolvedor)["citacoes"]
        self.assertEqual(len(citacoes), 1)
        self.assertEqual(citacoes[0]["classificacao"], "real")
        self.assertEqual(citacoes[0]["resolucao"], {"id_canonico": "2663361645"})

    def test_cnj_com_letra_de_ocr_e_quebra_de_linha_resolve_o_acordao_certo(self):
        for trecho, id_esperado in (
            (CNJ_COM_OCR_E_QUEBRA, 6688597116),
            (CNJ_COM_OCR_E_PONTO_PERDIDO, 2161632985),
        ):
            with self.subTest(trecho=trecho):
                citacoes = processar_documento("sintetico", trecho, self.resolvedor)["citacoes"]
                self.assertEqual(len(citacoes), 1)
                self.assertEqual(citacoes[0]["classificacao"], "real")
                self.assertEqual(citacoes[0]["resolucao"], {"id_canonico": str(id_esperado)})

    def test_cnj_com_letra_de_ocr_e_digito_trocado_continua_inventada(self):
        # O ajuste não pode tornar a leitura mais permissiva com identificadores adulterados.
        for trecho in (
            "RSE nº 7000768-37.\n2025.7.0O.0001 - RS",
            "RSE nº 7000769-37.\n2025.7.0O.0000 - RS",
            "APL 7000136-79.\n2023.7.00O000 9 - RS",
        ):
            with self.subTest(trecho=trecho):
                citacoes = processar_documento("sintetico", trecho, self.resolvedor)["citacoes"]
                self.assertEqual(len(citacoes), 1)
                self.assertNotEqual(citacoes[0]["classificacao"], "real")

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
