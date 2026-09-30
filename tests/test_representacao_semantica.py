"""Contrato semântico e decisões que não dependem do gabarito."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
        self.resolvedor.normativos = {"lei|Código de Processo Civil|373": [{"id": 42}]}
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
        zero = self.resolvedor.resolver_citacao(citacao("art. 999 do CPC", "lei"))
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

    def test_multiplos_normativos_continuam_incompletos(self):
        self.resolvedor.normativos["lei|Código de Processo Civil|373"].append({"id": 43})
        resultado = self.resolvedor.resolver_citacao(citacao("art. 373 do CPC", "lei"))
        self.assertEqual((resultado.classe, resultado.candidatos), ("incompleta", (42, 43)))

    def test_cobertura_explicita_para_familias_fora_do_catalogo(self):
        lei = self.resolvedor.resolver_citacao(citacao("art. 172 da Lei nº 9.504/1997", "lei"))
        tema = self.resolvedor.resolver_citacao(citacao("Tema 1234", "tema"))
        self.assertEqual((lei.classe, lei.cobertura, lei.motivo),
                         ("inventada", "fora_base", "lei_numerada_sem_mapeamento"))
        self.assertEqual((tema.classe, tema.cobertura), ("inventada", "fora_base"))

    def test_resultado_real_exige_id(self):
        with self.assertRaises(ValueError):
            ResultadoResolucao("real", None, "teste")


if __name__ == "__main__":
    unittest.main()
