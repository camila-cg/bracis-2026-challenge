"""Testes do gerador procedural de documentos sintéticos."""

import unittest

from gerar_documentos_sinteticos import (
    Anotacao,
    MODELO_GERADOR,
    Referencia,
    atribuir_splits,
    gerar_documento,
    transformar_superficie,
    validar_documento,
)


class GeradorSinteticoTest(unittest.TestCase):
    def setUp(self):
        self.texto = "Cabeçalho\n\nConforme REsp 1.741.784/PR, aplica-se o entendimento."
        inicio = self.texto.index("REsp")
        fim = self.texto.index(", aplica")
        self.anotacoes = [Anotacao(
            nivel="1", documento_id="origem", citacao_id="c1",
            inicio=inicio, fim=fim, trecho=self.texto[inicio:fim],
            tipo="jurisprudencia", classificacao="real", id_canonico="123",
        )]
        self.referencias = [Referencia(
            "Súmula 83 do STJ", "jurisprudencia", "real", "123", "sumula|STJ|83",
        )]

    def test_modelo_declara_que_nao_usa_llm(self):
        self.assertIn("sem LLM", MODELO_GERADOR)

    def test_split_e_deterministico_e_agrupado_por_origem(self):
        documentos = [f"doc_{indice:02d}" for indice in range(26)]
        self.assertEqual(atribuir_splits(documentos, 42), atribuir_splits(documentos, 42))
        self.assertEqual(
            {split: list(atribuir_splits(documentos, 42).values()).count(split)
             for split in ("train", "validation", "test")},
            {"train": 18, "validation": 4, "test": 4},
        )

    def test_transformacoes_preservam_algum_conteudo_e_recalculam_span(self):
        for modo in range(8):
            with self.subTest(modo=modo):
                transformado, _ = transformar_superficie("REsp nº 1.741.784/PR", modo)
                self.assertTrue(transformado)

    def test_documento_gerado_tem_spans_exatos(self):
        texto, anotacoes, _ = gerar_documento(
            self.texto, self.anotacoes, 1, self.referencias,
            {"83", "1741784"}, 12345,
        )
        self.assertGreaterEqual(len(anotacoes), 2)
        for anotacao in anotacoes:
            self.assertEqual(texto[anotacao["inicio"]:anotacao["fim"]], anotacao["trecho"])
        self.assertEqual(validar_documento("sintetico", texto, anotacoes, {123}), [])
        self.assertTrue(all(item.get("transformacao") for item in anotacoes))

    def test_perfil_sem_citacoes_remove_as_anotacoes(self):
        texto, anotacoes, metadados = gerar_documento(
            self.texto, self.anotacoes, 5, self.referencias,
            {"83", "1741784"}, 12345,
        )
        self.assertEqual(metadados["perfil"], "sem_citacoes")
        self.assertEqual(anotacoes, [])
        self.assertNotIn("REsp 1.741.784/PR", texto)

    def test_perfil_distratores_desloca_spans_sem_corromper_trechos(self):
        texto, anotacoes, metadados = gerar_documento(
            self.texto, self.anotacoes, 4, self.referencias,
            {"83", "1741784"}, 12345,
        )
        self.assertEqual(metadados["perfil"], "distratores")
        self.assertTrue(metadados["distratores"])
        self.assertEqual(validar_documento("sintetico", texto, anotacoes, {123}), [])


if __name__ == "__main__":
    unittest.main()
