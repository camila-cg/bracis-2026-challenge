"""Adaptação raw, curadoria e gate do holdout sintético em ZIP."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

import eda_utils as u
import pipeline as p
from validacao import holdout_zip as h
from validacao.teste_base_inteira import ids_duplicados

# Pisos medidos em 30/09/2026 com o ZIP saneado (alusões removidas e reais sobre
# duplicatas reclassificadas). Regrediu abaixo deles, a mudança piorou a
# generalização. Melhorou: suba o piso na mesma mudança.
PISO_SCORE_FINAL = 0.977
PISO_POR_NIVEL = {1: 1.007, 2: 0.962}


class BaseCanonica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (RAIZ / "data" / "desafio1_bracis.db").is_file():
            raise unittest.SkipTest("base canônica ausente em data/desafio1_bracis.db")
        if not h.ZIP.is_file():
            raise unittest.SkipTest("ZIP sintético ausente em datasets/")
        cls.con = u.conectar_base()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "con"):
            cls.con.close()


class Saneamento(BaseCanonica):
    def test_renomeia_documentos_para_o_formato_que_nivel_le(self):
        self.assertEqual(h.renomear_documento("jbsg26-1-01-e4d161dec7dd"), "gen_n1_001")
        self.assertEqual(h.renomear_documento("jbsg26-2-14-dbce42504b28"), "gen_n2_014")
        self.assertEqual(u.nivel(h.renomear_documento("jbsg26-2-14-dbce42504b28")), 2)
        with self.assertRaises(ValueError):
            h.renomear_documento("gen_n1_001")

    def test_ids_reais_ficam_numericos_e_existem_na_base(self):
        saneado = h.sanear(h.ZIP, self.con)
        reais = [linha for linha in saneado.gabarito if linha["classificacao"] == "real"]
        self.assertEqual(len(reais), 32)
        for linha in reais:
            with self.subTest(trecho=linha["trecho"][:60]):
                self.assertTrue(linha["id_canonico"].isdigit())
                existe = self.con.execute(
                    "SELECT COUNT(*) FROM documentos WHERE id = ?", (linha["id_canonico"],)
                ).fetchone()[0]
                self.assertEqual(existe, 1)
        self.assertEqual(saneado.relatorio["ids_remapeados"], 15)

    def test_offsets_e_nomes_continuam_consistentes(self):
        saneado = h.sanear(h.ZIP, self.con)
        self.assertEqual(len(saneado.textos), 26)
        for linha in saneado.gabarito:
            texto = saneado.textos[linha["documento_id"]]
            self.assertEqual(texto[int(linha["inicio"]):int(linha["fim"])], linha["trecho"])
            self.assertEqual(u.nivel(linha["documento_id"]), int(linha["nivel"]))

    def test_remove_alusoes_de_lei_sem_artigo_mas_mantem_a_jurisprudencia_descritiva(self):
        saneado = h.sanear(h.ZIP, self.con)
        self.assertEqual(len(saneado.relatorio["alusoes_removidas"]), 4)
        self.assertEqual(saneado.relatorio["citacoes_saneadas"], 67)
        self.assertFalse([
            linha for linha in saneado.gabarito
            if linha["tipo"] == "lei" and linha["classificacao"] == "incompleta"
        ])
        incompletas_jurisprudencia = [
            linha for linha in saneado.gabarito
            if linha["tipo"] == "jurisprudencia" and linha["classificacao"] == "incompleta"
        ]
        self.assertEqual(len(incompletas_jurisprudencia), 13)

    def test_reais_que_apontam_para_duplicatas_viram_incompleta_sem_id(self):
        duplicados = ids_duplicados(p.Resolvedor(self.con))
        saneado = h.sanear(h.ZIP, self.con, duplicados=duplicados)
        reclassificadas = saneado.relatorio["reclassificadas_por_duplicata"]
        self.assertEqual(len(reclassificadas), 5)
        reais = [linha for linha in saneado.gabarito if linha["classificacao"] == "real"]
        self.assertEqual(len(reais), 27)
        for linha in reais:
            self.assertNotIn(int(linha["id_canonico"]), duplicados)
        for linha in saneado.gabarito:
            if linha["trecho"] in reclassificadas:
                self.assertEqual((linha["classificacao"], linha["id_canonico"]), ("incompleta", ""))

    def test_sem_duplicados_o_gabarito_de_reais_nao_muda(self):
        saneado = h.sanear(h.ZIP, self.con)
        self.assertEqual(saneado.relatorio["reclassificadas_por_duplicata"], [])

    def test_manter_difusas_preserva_as_71_citacoes(self):
        saneado = h.sanear(h.ZIP, self.con, manter_difusas=True)
        self.assertEqual(saneado.relatorio["citacoes_saneadas"], 71)
        self.assertEqual(saneado.relatorio["alusoes_removidas"], [])

    def test_modo_raw_preserva_quantidades_e_classes_do_zip(self):
        saneado = h.sanear(
            h.ZIP,
            self.con,
            manter_difusas=True,
            duplicados=None,
        )
        por_classe = {
            classe: sum(
                linha["classificacao"] == classe for linha in saneado.gabarito
            )
            for classe in ("real", "inventada", "incompleta")
        }

        self.assertEqual(saneado.relatorio["citacoes_saneadas"], 71)
        self.assertEqual(por_classe, {"real": 32, "inventada": 22, "incompleta": 17})
        self.assertEqual(saneado.relatorio["alusoes_removidas"], [])
        self.assertEqual(saneado.relatorio["reclassificadas_por_duplicata"], [])
        self.assertEqual(saneado.relatorio["ids_remapeados"], 15)

    def test_gravar_escreve_textos_sem_crlf_e_gabarito_com_as_colunas_oficiais(self):
        saneado = h.sanear(h.ZIP, self.con)
        with tempfile.TemporaryDirectory() as pasta:
            gabarito = h.gravar(saneado, Path(pasta))
            cabecalho = gabarito.read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertEqual(cabecalho.split(","), h.COLUNAS)
            arquivos = sorted((Path(pasta) / "txt").glob("*.txt"))
            self.assertEqual(len(arquivos), 26)
            for arquivo in arquivos:
                self.assertNotIn(b"\r\n", arquivo.read_bytes())


class Gate(BaseCanonica):
    def test_score_do_holdout_nao_regride(self):
        with tempfile.TemporaryDirectory() as pasta, contextlib.redirect_stdout(io.StringIO()):
            resultado, _, _ = h.executar(Path(pasta))
        self.assertGreaterEqual(resultado["score_final"], PISO_SCORE_FINAL)
        for nivel, piso in PISO_POR_NIVEL.items():
            with self.subTest(nivel=nivel):
                obtido = {int(k): v["score"] for k, v in resultado["niveis"].items()}[nivel]
                self.assertGreaterEqual(obtido, piso)


class Comparacao(unittest.TestCase):
    def test_comparar_classifica_fn_fp_classe_e_id(self):
        gabarito = [
            {"documento_id": "d", "inicio": "0", "fim": "10", "trecho": "a", "classificacao": "real", "id_canonico": "1"},
            {"documento_id": "d", "inicio": "20", "fim": "30", "trecho": "b", "classificacao": "inventada", "id_canonico": ""},
            {"documento_id": "d", "inicio": "40", "fim": "50", "trecho": "c", "classificacao": "real", "id_canonico": "2"},
        ]
        documentos = [{"documento_id": "d", "citacoes": [
            {"inicio": 0, "fim": 10, "trecho": "a", "classificacao": "real", "resolucao": {"id_canonico": "9"}},
            {"inicio": 20, "fim": 30, "trecho": "b", "classificacao": "incompleta", "resolucao": None},
            {"inicio": 100, "fim": 110, "trecho": "z", "classificacao": "real", "resolucao": None},
        ]}]
        r = h.comparar(documentos, gabarito)
        self.assertEqual(len(r["pareadas"]), 2)
        self.assertEqual(len(r["id_errado"]), 1)
        self.assertEqual(len(r["classe_errada"]), 1)
        self.assertEqual(len(r["falsos_positivos"]), 1)
        self.assertEqual(len(r["falsos_negativos"]), 1)

    def test_iou_abaixo_de_meio_nao_pareia(self):
        gabarito = [{"documento_id": "d", "inicio": "0", "fim": "10", "trecho": "a", "classificacao": "real", "id_canonico": "1"}]
        documentos = [{"documento_id": "d", "citacoes": [
            {"inicio": 0, "fim": 30, "trecho": "a", "classificacao": "real", "resolucao": {"id_canonico": "1"}},
        ]}]
        r = h.comparar(documentos, gabarito)
        self.assertEqual(r["pareadas"], [])
        self.assertEqual(len(r["falsos_positivos"]), 1)
        self.assertEqual(len(r["falsos_negativos"]), 1)


if __name__ == "__main__":
    unittest.main()
