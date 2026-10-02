"""Contrato semântico e decisões que não dependem do gabarito."""

import sqlite3
import unittest
from unittest.mock import patch

import indice_utils as ind
from modelos import CitacaoNormalizada, ResultadoResolucao
from steps.buscar_na_base import Resolvedor
from steps.normalizar_citacao import normalizar_citacao


def citacao(trecho: str, familia: str) -> CitacaoNormalizada:
    return normalizar_citacao({"inicio": 7, "fim": 7 + len(trecho), "trecho": trecho}, familia)


class ExtracaoSemantica(unittest.TestCase):
    def test_acordao_preserva_span_e_extrai_campos(self):
        trecho = "REsp 1.741. 784/PR do STJ"
        c = citacao(trecho, "acordao")
        self.assertEqual((c.inicio, c.fim, c.trecho), (7, 7 + len(trecho), trecho))
        self.assertEqual((c.numero, c.classe, c.uf, c.tribunal),
                         ("1741784", "recursoespecial", "PR", "STJ"))
        self.assertIn("1.741.784", c.normalizado)

    def test_normativos_extraem_atributos_sem_consultar_base(self):
        lei = citacao("art. 373 do CPC", "lei")
        sumula = citacao("Súmula Vinculante 10", "sumula")
        self.assertEqual((lei.artigo, lei.diploma), ("373", "Código de Processo Civil"))
        self.assertEqual((sumula.numero, sumula.vinculante), ("10", True))

    def test_sv_exata_normaliza_como_sumula_vinculante_em_qualquer_caixa(self):
        for sigla in ("SV", "Sv", "sV", "sv"):
            with self.subTest(sigla=sigla):
                sumula = citacao(f"{sigla} 37", "sumula")
                self.assertEqual((sumula.numero, sumula.vinculante), ("37", True))


class DecisaoSemantica(unittest.TestCase):
    def setUp(self):
        self.resolvedor = Resolvedor.__new__(Resolvedor)
        self.resolvedor.con = None
        self.resolvedor.normativos = {
            "lei|Código de Processo Civil|373": [{"id": 42}],
            "lei_numero|lei|13105|2015|999": [{"id": 43}],
            "lei_numero|lei|13709|2018|7": [{"id": 44}],
            "lei_numero|lei|15999|2026|12": [{"id": 45}],
            "lei_numero|lei|400|2021|2": [{"id": 46}],
            "lei_numero|resolucao|400|2021|2": [{"id": 47}],
            "tema|STF|339": [{"id": 50}],
            "tema|STJ|1076": [{"id": 51}],
        }
        self.resolvedor.acordaos_por_numero = {
            "1741784": [
                {"id": 1, "classe": "REsp", "uf": "PR"},
                {"id": 2, "classe": "REsp", "uf": "PR"},
            ]
        }
        self.resolvedor.metadados_por_id = {
            1: ("STJ", "REsp", "PR"), 2: ("STJ", "REsp", "PR")
        }

    def test_consulta_normativa_zero_um_e_insuficiente(self):
        real = self.resolvedor.resolver_citacao(citacao("art. 373 do CPC", "lei"))
        zero = self.resolvedor.resolver_citacao(citacao("art. 998 do CPC", "lei"))
        incompleta = self.resolvedor.resolver_citacao(citacao("art. 373", "lei"))
        self.assertEqual((real.classe, real.id_canonico, real.candidatos), ("real", 42, (42,)))
        self.assertEqual((zero.classe, zero.id_canonico), ("inventada", None))
        self.assertEqual((incompleta.classe, incompleta.cobertura), ("incompleta", "nao_consultada"))

    def test_multiplos_acordaos_continuam_incompletos(self):
        resultado = self.resolvedor.resolver_citacao(citacao("REsp 1.741.784/PR do STJ", "acordao"))
        self.assertEqual((resultado.classe, resultado.candidatos), ("incompleta", (1, 2)))

    def test_um_acordao_compativel_e_real(self):
        self.resolvedor.metadados_por_id[2] = ("STJ", "REsp", "SC")
        resultado = self.resolvedor.resolver_citacao(citacao("REsp 1.741.784/PR do STJ", "acordao"))
        self.assertEqual((resultado.classe, resultado.id_canonico), ("real", 1))

    def test_fallback_fts_filtra_metadados_pelos_campos(self):
        self.resolvedor.acordaos_por_numero = {}
        with patch("steps.buscar_na_base.u.resolver_por_numero", return_value={1: 0}):
            resultado = self.resolvedor.resolver_citacao(citacao("REsp 1.741.784/SC do STJ", "acordao"))
        self.assertEqual((resultado.classe, resultado.caminho), ("inventada", "acordao_zero"))

    def test_fallback_sem_metadados_nao_confirma_classe_explicita(self):
        self.resolvedor.acordaos_por_numero = {}
        self.resolvedor.metadados_por_id = {1: ("TSE", "", None)}
        with patch("steps.buscar_na_base.u.resolver_por_numero", return_value={1: 0}):
            resultado = self.resolvedor.resolver_citacao(citacao(
                "Habeas Corpus nº 184-70.2012.6.20.0033 do TSE", "acordao"
            ))
        self.assertEqual((resultado.classe, resultado.caminho), ("incompleta", "acordao_ambigua"))
        self.assertEqual(resultado.candidatos, (1,))
        self.assertEqual(resultado.motivo, "metadados_nao_verificaveis")

    def test_multiplos_normativos_continuam_incompletos(self):
        self.resolvedor.normativos["lei|Código de Processo Civil|373"].append({"id": 43})
        resultado = self.resolvedor.resolver_citacao(citacao("art. 373 do CPC", "lei"))
        self.assertEqual((resultado.classe, resultado.candidatos), ("incompleta", (42, 43)))

    def test_dispositivo_novo_resolve_sem_id_pre_configurado(self):
        casos = (
            ("art. 999 do CPC", 43),
            ("art. 7º da LGPD", 44),
            ("art. 12 da Lei nº 15.999/2026", 45),
        )
        for trecho, id_canonico in casos:
            with self.subTest(trecho=trecho):
                resultado = self.resolvedor.resolver_citacao(citacao(trecho, "lei"))
                self.assertEqual((resultado.classe, resultado.id_canonico), ("real", id_canonico))

    def test_lei_federal_desconhecida_consulta_base_antes_de_rejeitar(self):
        nova = citacao("art. 12 da Lei nº 15.999/2026", "lei")
        self.assertIsNone(nova.identidade_legal_valida)

        ausente = citacao("art. 13 da Lei nº 15.999/2026", "lei")
        resultado = self.resolvedor.resolver_citacao(ausente)
        self.assertEqual(resultado.classe, "inventada")

    def test_especie_do_ato_impede_colisao_de_numero_e_ano(self):
        lei = self.resolvedor.resolver_citacao(
            citacao("art. 2 da Lei nº 400/2021", "lei")
        )
        resolucao = self.resolvedor.resolver_citacao(
            citacao("art. 2 da Resolução CNJ nº 400/2021", "lei")
        )
        self.assertEqual((lei.classe, lei.id_canonico), ("real", 46))
        self.assertEqual((resolucao.classe, resolucao.id_canonico), ("real", 47))

    def test_tema_indexado_pode_ser_real(self):
        tema_stf = self.resolvedor.resolver_citacao(citacao("Tema 339/STF", "tema"))
        tema_stj = self.resolvedor.resolver_citacao(citacao("Tema 1.076 do STJ", "tema"))
        self.assertEqual((tema_stf.classe, tema_stf.id_canonico), ("real", 50))
        self.assertEqual((tema_stj.classe, tema_stj.id_canonico), ("real", 51))

    def test_tema_sem_tribunal_resolve_apenas_quando_numero_e_unico(self):
        unico = self.resolvedor.resolver_citacao(citacao("Tema 339", "tema"))
        self.assertEqual((unico.classe, unico.id_canonico), ("real", 50))

        self.resolvedor.normativos["tema|STJ|339"] = [{"id": 52}]
        ambiguo = self.resolvedor.resolver_citacao(citacao("Tema 339", "tema"))
        self.assertEqual((ambiguo.classe, ambiguo.candidatos), ("incompleta", (50, 52)))

    def test_tema_ausente_da_base_continua_inventado(self):
        tema = self.resolvedor.resolver_citacao(citacao("Tema 1234", "tema"))
        self.assertEqual((tema.classe, tema.cobertura), ("inventada", "indexada"))

    def test_indice_canonico_inclui_temas(self):
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE documentos (documento_id TEXT, id INTEGER, "
            "natureza TEXT, tribunal TEXT, texto TEXT)"
        )
        con.executemany(
            "INSERT INTO documentos VALUES (?, ?, ?, ?, ?)",
            [
                (
                    "lei-nova-art-12", 69, "dispositivo", None,
                    "Artigo 12 da Lei nº 15.999/2026\nArt. 12. Texto canônico.",
                ),
                ("tema-stf-339", 70, "tema", "STF", "Tema 339 - fundamentação"),
                ("tema-stj-1076", 71, "tema", None, "Tema repetitivo 1.076 do STJ"),
            ],
        )
        indice, sem_resolucao = ind.construir_indice_normativos(con)
        self.assertEqual(sem_resolucao, [])
        self.assertEqual(indice["lei_numero|lei|15999|2026|12"][0]["id"], 69)
        self.assertEqual(indice["tema|STF|339"][0]["id"], 70)
        self.assertEqual(indice["tema|STJ|1076"][0]["id"], 71)

    def test_indice_canonico_extrai_sumulas_do_titulo_sem_ids_fixos(self):
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE documentos (documento_id TEXT, id INTEGER, tribunal TEXT, "
            "ano INTEGER, relator TEXT, natureza TEXT, tipo TEXT, texto TEXT, texto_len INTEGER)"
        )
        con.executemany(
            "INSERT INTO documentos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "sumula-tst-999", 72, "TST", None, None, "sumula", "sumula",
                    "Súmula n. 999\nEnunciado novo.", 30,
                ),
                (
                    "sumula-vinculante-42", 73, "STF", None, None, "sumula", "sumula",
                    "Súmula Vinculante nº 42 do STF\nEnunciado. (SÚMULA 42, STF)", 61,
                ),
            ],
        )

        indice, sem_resolucao = ind.construir_indice_normativos(con)

        self.assertEqual(sem_resolucao, [])
        self.assertEqual(indice["sumula|TST|999"][0]["id"], 72)
        self.assertEqual(indice["sumula|STF|VINCULANTE-42"][0]["id"], 73)
        self.assertNotIn("sumula|STF|42", indice)
        resolvedor = Resolvedor(con)
        self.assertEqual(resolvedor.sumula("Súmula 999 do TST")[:2], ("real", 72))
        self.assertEqual(resolvedor.sumula("Súmula Vinculante 42")[:2], ("real", 73))

    def test_cobertura_explicita_para_lei_fora_do_catalogo(self):
        lei = self.resolvedor.resolver_citacao(citacao("art. 172 da Lei nº 9.504/1997", "lei"))
        self.assertEqual((lei.classe, lei.cobertura, lei.motivo),
                         ("inventada", "fora_base", "lei_numerada_sem_mapeamento"))

    def test_resultado_real_exige_id(self):
        with self.assertRaises(ValueError):
            ResultadoResolucao("real", None, "teste")


if __name__ == "__main__":
    unittest.main()
