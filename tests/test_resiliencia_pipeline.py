"""Regressões da tolerância a falhas por arquivo, modelo e citação."""

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pipeline import (
    gravar_log_erros,
    processar_arquivo,
    processar_documento,
)
from steps.salvar_submissao import gravar_saida


class ResilienciaPorArquivo(unittest.TestCase):
    def test_arquivo_invalido_mantem_linha_vazia_e_nao_interrompe_o_lote(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            invalido = raiz / "invalido.txt"
            valido = raiz / "valido.txt"
            invalido.write_bytes(b"linha 1\r\nlinha 2")
            valido.write_text("texto valido", encoding="utf-8")
            erros = []

            def processar_valido(documento_id, *_args, **_kwargs):
                return {"documento_id": documento_id, "citacoes": []}

            with patch("pipeline.processar_documento", side_effect=processar_valido):
                documentos = [
                    processar_arquivo(
                        arquivo,
                        Mock(),
                        erros_processamento=erros,
                    )
                    for arquivo in (invalido, valido)
                ]

            csv_saida = gravar_saida(documentos, raiz / "saida")
            with csv_saida.open(encoding="utf-8", newline="") as arquivo:
                linhas = list(csv.DictReader(arquivo))

        self.assertEqual([linha["documento_id"] for linha in linhas], ["invalido", "valido"])
        self.assertEqual(linhas[0]["citacoes"], "-")
        self.assertEqual(erros[0]["etapa"], "leitura")
        self.assertEqual(erros[0]["acao"], "documento_emitido_sem_citacoes")

    def test_log_jsonl_nao_inclui_texto_do_documento(self):
        erros = []
        with tempfile.TemporaryDirectory() as tmp:
            arquivo = Path(tmp) / "segredo.txt"
            arquivo.write_bytes(b"conteudo\r\nconfidencial")
            processar_arquivo(arquivo, Mock(), erros_processamento=erros)
            caminho = gravar_log_erros(
                erros,
                Path(tmp) / "logs",
                datetime(2026, 10, 1, 12, 0, 0),
            )
            registro = json.loads(caminho.read_text(encoding="utf-8"))

        self.assertEqual(registro["arquivo"], "segredo.txt")
        self.assertNotIn("conteudo", json.dumps(registro, ensure_ascii=False))
        self.assertNotIn("confidencial", json.dumps(registro, ensure_ascii=False))


class ResilienciaPorEtapa(unittest.TestCase):
    def test_citacao_invalida_e_descartada_sem_perder_as_demais(self):
        texto = "art. 5º da CF"
        candidatos = [
            {"tipo": "lei", "inicio": 0, "fim": 1},
            {"tipo": "lei", "inicio": 0, "fim": len(texto), "trecho": texto},
        ]
        resolvedor = Mock()
        resolvedor.resolver_citacao.return_value = SimpleNamespace(
            caminho="lei_real",
            classe="real",
            id_canonico=123,
            cobertura="indexada",
            motivo=None,
        )
        erros = []

        with patch("pipeline.extrair", return_value=candidatos), patch(
            "pipeline.filtrar_distratores", side_effect=lambda itens, _texto: itens
        ):
            documento = processar_documento(
                "doc",
                texto,
                resolvedor,
                erros_processamento=erros,
            )

        self.assertEqual(len(documento["citacoes"]), 1)
        self.assertEqual(documento["citacoes"][0]["resolucao"], {"id_canonico": "123"})
        self.assertEqual(erros[0]["acao"], "citacao_descartada")
        self.assertEqual(erros[0]["etapa"], "roteamento")

    def test_falha_do_detector_faz_fallback_para_regras(self):
        texto = "art. 5º da CF"
        candidato = {"tipo": "lei", "inicio": 0, "fim": len(texto), "trecho": texto}
        detector = Mock()
        detector.detectar.side_effect = RuntimeError("modelo indisponível")
        resolvedor = Mock()
        resolvedor.resolver_citacao.return_value = SimpleNamespace(
            caminho="lei_real",
            classe="real",
            id_canonico=123,
            cobertura="indexada",
            motivo=None,
        )
        erros = []

        with patch("pipeline.extrair", return_value=[candidato]), patch(
            "pipeline.filtrar_distratores", side_effect=lambda itens, _texto: itens
        ):
            documento = processar_documento(
                "doc",
                texto,
                resolvedor,
                detector_bio=detector,
                erros_processamento=erros,
            )

        self.assertEqual(len(documento["citacoes"]), 1)
        self.assertNotIn("eventos_deteccao", documento)
        self.assertEqual(erros[0]["etapa"], "deteccao_modelo")
        self.assertEqual(erros[0]["acao"], "fallback_para_regras")


if __name__ == "__main__":
    unittest.main()
