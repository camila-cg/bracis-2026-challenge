"""Contrato do card 1: localizar citações nos .txt e remover distratores.

Execute da raiz: python -m unittest tests/test_card_01_extracao.py -v
Os testes usam textos pequenos e não dependem do gabarito nem da base canônica.
"""

import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import detectar_citacoes as detector
from steps.extrair_candidatos import extrair
from steps.filtrar_distratores import analisar_contexto, e_distrator, filtrar_distratores, motivo_distrator


def candidatos(texto):
    return filtrar_distratores(extrair(texto), texto)


class OffsetsEQuebrasDeLinha(unittest.TestCase):
    def test_normalizacao_preserva_comprimento_e_offsets_unicode(self):
        texto = "🔎 Prévia:\tREsp 1.741.784/PR\nart. 373 do CPC"
        normalizado = detector.criar_texto_normalizado(texto)
        self.assertEqual(len(normalizado), len(texto))
        for token in detector.criar_tokens(texto):
            self.assertEqual(token["original"], texto[token["inicio"]:token["fim"]])
            self.assertEqual(token["normalizado"], normalizado[token["inicio"]:token["fim"]])

    def test_quebras_de_linha_e_acentos_mantem_trecho_original(self):
        casos = [
            "Segundo o REsp\n1.741.784/PR, decidiu-se.",
            "Segundo o REsp\r\n1.741.784/PR, decidiu-se.",
            "Aplica-se a Súmula\t83 do STJ.",
            "Aplica-se o art.\n373 do CPC.",
        ]
        for texto in casos:
            with self.subTest(texto=repr(texto)):
                encontrados = candidatos(texto)
                self.assertTrue(encontrados)
                for item in encontrados:
                    self.assertEqual(item["trecho"], texto[item["inicio"]:item["fim"]])

    def test_citacoes_separadas_por_ponto_viram_spans_distintos(self):
        texto = "Conforme REsp 1.741.784/PR. Também incide a Lei 8.078/1990."
        encontrados = candidatos(texto)
        self.assertEqual([item["tipo"] for item in encontrados], ["jurisprudencia", "lei"])
        self.assertLessEqual(encontrados[0]["fim"], encontrados[1]["inicio"])

    def test_duas_citacoes_na_mesma_frase_nao_sao_fundidas(self):
        texto = "REsp 1.741.784/PR, HC 123456 do STF"
        self.assertEqual(
            [item["trecho"] for item in candidatos(texto)],
            ["REsp 1.741.784/PR", "HC 123456 do STF"],
        )

    def test_span_e_exclusivo_no_fim_e_independe_do_prefixo(self):
        citacao = "REsp 1.741.784/PR"
        for prefixo in ("", "Antes: ", "🔎\n\tAntes: "):
            with self.subTest(prefixo=repr(prefixo)):
                texto = prefixo + citacao + "."
                encontrado = next(item for item in candidatos(texto) if item["tipo"] == "jurisprudencia")
                self.assertEqual((encontrado["inicio"], encontrado["fim"]), (len(prefixo), len(prefixo) + len(citacao)))
                self.assertEqual(texto[encontrado["inicio"]:encontrado["fim"]], citacao)

    def test_texto_sem_citacao_nao_gera_candidato(self):
        for texto in ("", "Discussão sobre fatos e provas sem referência normativa."):
            with self.subTest(texto=texto):
                self.assertEqual(candidatos(texto), [])


class TiposDeCitacao(unittest.TestCase):
    def test_jurisprudencia_em_formatos_representativos(self):
        for trecho in ("REsp 1.741.784/PR", "HC 123456 do STF", "Agravo de Instrumento 123456"):
            with self.subTest(trecho=trecho):
                encontrados = candidatos("Conforme " + trecho + ", há precedente.")
                self.assertTrue(any(item["tipo"] == "jurisprudencia" for item in encontrados))

    def test_sumulas_comuns_e_vinculantes(self):
        for trecho in ("Súmula 83 do STJ", "Súmula Vinculante 10 do STF", "sum. 331 do TST"):
            with self.subTest(trecho=trecho):
                encontrados = candidatos("Aplica-se a " + trecho + ".")
                self.assertTrue(any(item["tipo"] == "jurisprudencia" for item in encontrados))

    def test_lei_artigo_e_paragrafo(self):
        for trecho in ("Lei 8.078/1990", "art. 373 do CPC", "artigo 5º da Constituição", "§ 1º do art. 489 do CPC"):
            with self.subTest(trecho=trecho):
                encontrados = candidatos("Aplica-se " + trecho + ".")
                self.assertTrue(any(item["tipo"] == "lei" for item in encontrados))

    def test_marcador_isolado_nao_basta(self):
        for texto in ("O recurso foi apresentado.", "A lei foi debatida.", "O artigo é relevante."):
            with self.subTest(texto=texto):
                self.assertEqual(candidatos(texto), [])


class FiltrosDeDistratores(unittest.TestCase):
    def test_valor_da_causa_sem_cifrao_e_filtrado_pelo_contexto(self):
        for texto, quantia in (
            ("Valor da causa: 10.000,00", "10.000,00"),
            ("Valor atribuído à causa de 25000,00", "25000,00"),
            ("Dá-se à causa o valor de 5000", "5000"),
        ):
            with self.subTest(texto=texto):
                inicio = texto.index(quantia)
                item = {"inicio": inicio, "fim": inicio + len(quantia), "trecho": quantia}
                self.assertEqual(filtrar_distratores([item], texto), [])

    def test_citacao_na_mesma_linha_do_valor_e_preservada(self):
        texto = "Valor da causa: 10.000,00. Conforme REsp 1.741.784/PR."
        citacao = "REsp 1.741.784/PR"
        inicio = texto.index(citacao)
        item = {"inicio": inicio, "fim": inicio + len(citacao), "trecho": citacao}
        self.assertEqual(filtrar_distratores([item], texto), [item])

    def test_valor_da_causa_em_linha_anterior_nao_afeta_citacao(self):
        texto = "Valor da causa: 10.000,00\nREsp 1.741.784/PR"
        citacao = "REsp 1.741.784/PR"
        inicio = texto.index(citacao)
        item = {"inicio": inicio, "fim": inicio + len(citacao), "trecho": citacao}
        self.assertEqual(filtrar_distratores([item], texto), [item])

    def test_valor_da_causa_com_digito_corrompido_por_ocr_e_filtrado(self):
        # "1O.741.784,00": o dígito "1" isolado virou a letra "O" (OCR), colado a dígitos reais.
        texto = "Valor da causa: R$ 1O.741.784,00"
        quantia = "1O.741.784,00"
        inicio = texto.index(quantia)
        item = {"inicio": inicio, "fim": inicio + len(quantia), "trecho": quantia}
        self.assertEqual(filtrar_distratores([item], texto), [])

    def test_valor_da_causa_com_quantia_na_linha_seguinte_e_filtrado(self):
        texto = "Valor da causa:\nR$ 10.000,00"
        quantia = "10.000,00"
        inicio = texto.index(quantia)
        item = {"inicio": inicio, "fim": inicio + len(quantia), "trecho": quantia}
        self.assertEqual(filtrar_distratores([item], texto), [])

    def test_valor_da_causa_em_linha_isolada_de_prosa_nao_afeta_citacao(self):
        # "valor da causa" mencionado em prosa (sem ":" nem quantia na própria linha)
        # não deve puxar uma citação real na linha seguinte para dentro do campo.
        texto = "O juiz fixou o valor da causa\ne determinou a citação do réu conforme REsp 1.741.784/PR."
        citacao = "REsp 1.741.784/PR"
        inicio = texto.index(citacao)
        item = {"inicio": inicio, "fim": inicio + len(citacao), "trecho": citacao}
        self.assertEqual(filtrar_distratores([item], texto), [item])

    def test_oab_folhas_e_valor_monetario_sao_filtrados(self):
        for trecho in ("OAB 12345 REsp 123456", "fls. 12 REsp 123456", "Valor da causa R$ 123456 REsp 123456"):
            with self.subTest(trecho=trecho):
                self.assertTrue(e_distrator({"inicio": 500, "trecho": trecho}))

    def test_cabecalho_do_proprio_processo_e_filtrado(self):
        for trecho in ("Processo 123456", "Autos 123456", "Relator: Fulano de Tal", "Reclamante: Fulano de Tal"):
            with self.subTest(trecho=trecho):
                self.assertTrue(e_distrator({"inicio": 0, "fim": len(trecho), "trecho": trecho}, trecho))

    def test_cabecalho_longo_e_filtrado_pelo_rotulo(self):
        prefixo = "PODER JUDICIÁRIO\n" * 30
        trecho = "Processo 123456"
        texto = prefixo + trecho + "\n\nACÓRDÃO\nConforme REsp 1.741.784/PR."
        inicio = len(prefixo)
        self.assertGreater(inicio, 400)
        item = {"inicio": inicio, "fim": inicio + len(trecho), "trecho": trecho}
        self.assertEqual(motivo_distrator(item, texto), "campo_processo_principal")

    def test_numero_principal_repetido_no_corpo_e_preservado(self):
        texto = "Processo 123456\nRelator: Fulano\n\nACÓRDÃO\nConforme Processo 123456, decide-se."
        trecho = "Processo 123456"
        inicio = texto.rindex(trecho)
        item = {"inicio": inicio, "fim": inicio + len(trecho), "trecho": trecho}
        self.assertEqual(analisar_contexto(texto).processo_principal, "123456")
        self.assertIsNone(motivo_distrator(item, texto))

    def test_transicao_para_corpo_sem_titulo(self):
        texto = "Autos 123456\nRecorrente: Maria\n\nConforme Processo 654321, há precedente."
        trecho = "Processo 654321"
        inicio = texto.index(trecho)
        item = {"inicio": inicio, "fim": inicio + len(trecho), "trecho": trecho}
        self.assertIsNone(motivo_distrator(item, texto))

    def test_palavra_de_parte_sem_rotulo_no_corpo_nao_vira_campo(self):
        texto = "Autos 123456\nRecorrente: Maria\n\nRecorrido o art. 373 do CPC, decide-se."
        trecho = "Recorrido o art. 373 do CPC"
        inicio = texto.index(trecho)
        item = {"inicio": inicio, "fim": inicio + len(trecho), "trecho": trecho}
        self.assertIsNone(motivo_distrator(item, texto))

    def test_span_que_reune_campos_de_metadados(self):
        texto = "Processo 123456\nReclamada: Maria\nRelator: Fulano\n\nACÓRDÃO\n"
        trecho = "Reclamada: Maria\nRelator: Fulano"
        inicio = texto.index(trecho)
        item = {"inicio": inicio, "fim": inicio + len(trecho), "trecho": trecho}
        self.assertEqual(motivo_distrator(item, texto), "campo_reclamada")

    def test_distrator_no_inicio_nao_remove_citacao_posterior(self):
        for prefixo, citacao in (
            ("Processo 123456. Conforme ", "REsp 1.741.784/PR"),
            ("OAB 12345. Conforme ", "REsp 1.741.784/PR"),
            ("Valor da causa R$ 1.000,00. Conforme ", "art. 373 do CPC"),
        ):
            with self.subTest(prefixo=prefixo):
                texto = prefixo + citacao + "."
                encontrados = candidatos(texto)
                self.assertEqual([item["trecho"] for item in encontrados], [citacao])

    def test_citacao_apos_cabecalho_e_preservada(self):
        item = {"inicio": 500, "trecho": "Processo REsp 1.741.784/PR"}
        self.assertFalse(e_distrator(item))
        self.assertEqual(filtrar_distratores([item]), [item])

    def test_filtro_nao_descarta_referencia_valida_no_inicio(self):
        item = {"inicio": 0, "trecho": "REsp 1.741.784/PR"}
        self.assertFalse(e_distrator(item))

    def test_filtragem_nao_altera_offsets_dos_sobreviventes(self):
        texto = "Processo 123456\n\nACÓRDÃO\nConforme REsp 1.741.784/PR."
        falso_trecho = "Processo 123456"
        valido_trecho = "REsp 1.741.784/PR"
        falso = {"inicio": 0, "fim": len(falso_trecho), "trecho": falso_trecho}
        inicio = texto.index(valido_trecho)
        valido = {"inicio": inicio, "fim": inicio + len(valido_trecho), "trecho": valido_trecho}
        self.assertEqual(filtrar_distratores([falso, valido], texto), [valido])


if __name__ == "__main__":
    unittest.main()
