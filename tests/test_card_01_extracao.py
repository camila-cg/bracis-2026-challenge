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
from steps.rotear_citacao import rotear


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

    def test_numero_de_outra_frase_nao_alonga_citacao_completa(self):
        texto = "Tema 339/STF. O valor 241.907 foi lançado na estimativa."
        self.assertEqual(
            [item["trecho"] for item in candidatos(texto)],
            ["Tema 339/STF"],
        )

    def test_barreira_numerica_preserva_abreviacoes_da_citacao(self):
        casos = (
            ("Aplica-se o art. 5º da Constituição.", "art. 5º da Constituição"),
            ("Aplica-se o Tema n. 27.", "Tema n. 27"),
            ("Conforme a Lei n. 8.078/1990.", "Lei n. 8.078/1990"),
            (
                "REsp 1.741.784/PR, rel. Min. Assusete Magalhães.",
                "REsp 1.741.784/PR, rel. Min. Assusete Magalhães",
            ),
        )
        for texto, esperado in casos:
            with self.subTest(texto=texto):
                self.assertEqual(
                    [item["trecho"] for item in candidatos(texto)],
                    [esperado],
                )

    def test_barreira_numerica_preserva_numero_ocr_quebrado_em_linha(self):
        citacao = "AgInt no R.Esp. n.º l.8O4.\n55O—RS"
        texto = "Conforme " + citacao + ", manteve-se o acórdão."
        self.assertEqual(
            [item["trecho"] for item in candidatos(texto)],
            [citacao],
        )

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
        for trecho in (
            "REsp 1.741.784/PR",
            "HC 123456 do STF",
            "Agravo de Instrumento 123456",
            "Mandado de Segurança 32714",
        ):
            with self.subTest(trecho=trecho):
                encontrados = candidatos("Conforme " + trecho + ", há precedente.")
                self.assertTrue(any(item["tipo"] == "jurisprudencia" for item in encontrados))

    def test_sumulas_comuns_e_vinculantes(self):
        for trecho in ("Súmula 83 do STJ", "Súmula Vinculante 10 do STF", "sum. 331 do TST"):
            with self.subTest(trecho=trecho):
                encontrados = candidatos("Aplica-se a " + trecho + ".")
                self.assertTrue(any(item["tipo"] == "jurisprudencia" for item in encontrados))

    def test_sv_exata_sem_aproximacao_aceita_qualquer_caixa(self):
        for sigla in ("SV", "Sv", "sV", "sv"):
            with self.subTest(sigla=sigla):
                texto = f"Aplicam-se o Tema 1.076/STJ e a {sigla} 37."
                encontrados = candidatos(texto)
                self.assertEqual(
                    [item["trecho"] for item in encontrados],
                    ["Tema 1.076/STJ", f"{sigla} 37"],
                )
                self.assertEqual(rotear(encontrados[1]), "sumula")

    def test_sv_exige_token_exato_e_numero_no_token_seguinte(self):
        for trecho in ("SV", "SV do STF", "SV nº 37", "S V 37", "SVX 37", "5V 37"):
            with self.subTest(trecho=trecho):
                self.assertEqual(candidatos("Aplicou-se " + trecho + "."), [])

    def test_tema_exige_numero_apos_conector_juridico(self):
        casos = (
            "Tema 27",
            "Tema nº 27",
            "Tema n. 27",
            "Tema número 27",
            "Tema repetitivo 27",
            "Tema repetitivo nº 27",
            "Tema de repercussão geral nº 27",
            "Tema da repercussão geral 27",
            "Temã 2.680 da repercussão geral",
        )
        for trecho in casos:
            with self.subTest(trecho=trecho):
                self.assertEqual(
                    [item["trecho"] for item in candidatos(trecho)],
                    [trecho],
                )

    def test_tema_discursivo_numerado_nao_vira_citacao(self):
        for trecho in (
            "Tema em debate: 27",
            "Tema da reunião: 27",
            "Tema relevante 27",
            "Tema sem número",
        ):
            with self.subTest(trecho=trecho):
                self.assertEqual(candidatos(trecho), [])

    def test_lei_artigo_e_paragrafo(self):
        for trecho in ("Lei 8.078/1990", "art. 373 do CPC", "artigo 5º da Constituição", "§ 1º do art. 489 do CPC"):
            with self.subTest(trecho=trecho):
                encontrados = candidatos("Aplica-se " + trecho + ".")
                self.assertTrue(any(item["tipo"] == "lei" for item in encontrados))

    def test_novo_diploma_coordenado_inicia_outro_span(self):
        casos = (
            (
                "Cód. de Proc. Penal-Militar, art. 5º, inc. II e o "
                "Decr. 9.83O, de 1O/6/2O19",
                [
                    "Cód. de Proc. Penal-Militar, art. 5º, inc. II",
                    "Decr. 9.83O, de 1O/6/2O19",
                ],
            ),
            (
                "Cód. de Proc. Penal-Militar, art. 5º, inciso II bem como o "
                "Decreto 9.830/2019",
                [
                    "Cód. de Proc. Penal-Militar, art. 5º, inciso II",
                    "Decreto 9.830/2019",
                ],
            ),
            (
                "Cód. de Proc. Penal-Militar, art. 5º, inciso II; "
                "Decreto 9.830/2019",
                [
                    "Cód. de Proc. Penal-Militar, art. 5º, inciso II",
                    "Decreto 9.830/2019",
                ],
            ),
        )
        for texto, esperado in casos:
            with self.subTest(texto=texto):
                self.assertEqual(
                    [item["trecho"] for item in candidatos(texto)],
                    esperado,
                )

    def test_estruturas_do_mesmo_diploma_permanecem_no_mesmo_span(self):
        casos = (
            "arts. 5º e 7º da Constituição",
            "art. 5º e art. 7º da Constituição",
            "art. 5º, incisos I e II do Código Penal",
            "art. 5º, alínea a, do Código Penal",
        )
        for texto in casos:
            with self.subTest(texto=texto):
                self.assertEqual(
                    [item["trecho"] for item in candidatos(texto)],
                    [texto],
                )

    def test_marcador_isolado_nao_basta(self):
        for texto in ("O recurso foi apresentado.", "A lei foi debatida.", "O artigo é relevante."):
            with self.subTest(texto=texto):
                self.assertEqual(candidatos(texto), [])

    def test_ms_ambiguo_e_classe_processual_mas_nao_marcador_principal(self):
        texto = "MS 32.714/MS, Plenário do STF, rel. Min. Rosa Weber, julgamento em 6.11.2014"
        tokens = detector.criar_tokens(texto)
        registros = detector.detectar_marcadores_e_indicios(texto, tokens)

        self.assertFalse(any(
            registro["tipo_registro"] == "marcador_principal"
            and registro.get("marcador") == "ms"
            for registro in registros
        ))
        self.assertTrue(any(
            registro["tipo_registro"] == "classe_processual"
            and registro.get("classe") == "mandadodeseguranca"
            and registro["pontos"] == 1
            for registro in registros
        ))

        encontrados = candidatos(texto)
        self.assertEqual([item["trecho"] for item in encontrados], [texto])

    def test_ai_exata_com_identificador_e_classe_processual(self):
        for texto in (
            "AI nº 1024567-34.2023.8.26.0000/TJSP",
            "AI 123456/MG",
            "ai 123456",
        ):
            with self.subTest(texto=texto):
                tokens = detector.criar_tokens(texto)
                registros = detector.detectar_marcadores_e_indicios(texto, tokens)

                self.assertFalse(any(
                    registro["tipo_registro"] == "marcador_principal"
                    and registro.get("marcador") == "ai"
                    for registro in registros
                ))
                self.assertTrue(any(
                    registro["tipo_registro"] == "classe_processual"
                    and registro.get("classe") == "agravodeinstrumento"
                    for registro in registros
                ))
                self.assertEqual(
                    [(item["tipo"], item["trecho"]) for item in candidatos(texto)],
                    [("jurisprudencia", texto)],
                )

    def test_ai_ambigua_sem_identificador_nao_vira_citacao(self):
        for texto in (
            "AI",
            "AI foi utilizada no processo.",
            "AI 2024",
            "AI nº 37",
            "Aí 123456 começou a discussão.",
            "O sistema de AI 123456 foi atualizado.",
        ):
            with self.subTest(texto=texto):
                self.assertEqual(candidatos(texto), [])

    def test_ms_geografico_nao_vira_citacao(self):
        for texto in (
            "Investimentos no MS 2024 cresceram.",
            "A reunião aconteceu em Campo Grande/MS.",
        ):
            with self.subTest(texto=texto):
                self.assertEqual(candidatos(texto), [])

    def test_ms_geografico_em_citacao_de_lei_nao_vira_classe_processual(self):
        texto = "A lei do Estado de MS nº 123 foi regulamentada."
        tokens = detector.criar_tokens(texto)
        registros = detector.detectar_marcadores_e_indicios(texto, tokens)

        self.assertFalse(any(
            registro["tipo_registro"] == "classe_processual"
            for registro in registros
        ))
        self.assertEqual(
            [item["tipo"] for item in candidatos(texto)],
            ["lei"],
        )

    def test_apelacao_criminal_e_artigo_formam_citacoes_separadas(self):
        jurisprudencia = "Ap. Crim. 7000424—61.2022.7.00.0000/DF, STM"
        lei = "art. 290-B do Código Penal Militar"
        texto = f"{jurisprudencia} e o suposto {lei}"

        encontrados = candidatos(texto)

        self.assertEqual(
            [(item["tipo"], item["trecho"]) for item in encontrados],
            [("jurisprudencia", jurisprudencia), ("lei", lei)],
        )

    def test_ro_e_ro_eleitoral_sao_classes_ambiguas_com_numero(self):
        casos = (
            ("RO 1984-03.2014.6.08.0000, TSE", "recursoordinario"),
            ("RO-El 0600710-22.2022.6.12.0000, TSE", "recursoordinarioeleitoral"),
        )

        for texto, classe in casos:
            with self.subTest(texto=texto):
                tokens = detector.criar_tokens(texto)
                registros = detector.detectar_marcadores_e_indicios(texto, tokens)

                self.assertFalse(any(
                    registro["tipo_registro"] == "marcador_principal"
                    and registro.get("marcador") in {"ro", "ro-el"}
                    for registro in registros
                ))
                self.assertTrue(any(
                    registro["tipo_registro"] == "classe_processual"
                    and registro.get("classe") == classe
                    for registro in registros
                ))
                self.assertEqual(
                    [(item["tipo"], item["trecho"]) for item in candidatos(texto)],
                    [("jurisprudencia", texto)],
                )

    def test_ro_geografico_nao_vira_citacao(self):
        for texto in ("Investimentos em RO 2024 cresceram.", "A obra ocorreu em RO 123."):
            with self.subTest(texto=texto):
                self.assertEqual(candidatos(texto), [])

    def test_ro_e_ro_eleitoral_na_mesma_frase_nao_sao_fundidos(self):
        primeira = "RO 1984-03.2014.6.08.0000, TSE"
        segunda = "RO-El 0600710-22.2022.6.12.0000, TSE"
        texto = f"{primeira} e {segunda}"

        self.assertEqual(
            [(item["tipo"], item["trecho"]) for item in candidatos(texto)],
            [("jurisprudencia", primeira), ("jurisprudencia", segunda)],
        )

    def test_apelacao_exige_correspondencia_exata(self):
        texto = "Apelação 0027419-08, da Câmara de Família"
        tokens = detector.criar_tokens(texto)
        registros = detector.detectar_marcadores_e_indicios(texto, tokens)

        self.assertTrue(any(
            registro["tipo_registro"] == "marcador_principal"
            and registro.get("marcador") == "apelacao"
            and registro.get("similaridade") == 1.0
            for registro in registros
        ))
        self.assertEqual(
            [item["tipo"] for item in candidatos(texto)],
            ["jurisprudencia"],
        )

    def test_apelante_nao_e_confundido_com_apelacao(self):
        texto = (
            "9293337-24.2018.7.15.8725\n"
            "Apelante: TRANSPORTES MARAJÓ EIRELI\n"
            "Apelado: Ministério Público Militar\n"
            "Protocolo nº 2023.1475691"
        )

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

    def test_titulo_de_secao_numerada_e_reconhecido(self):
        # "I — RELATÓRIO" (numeração + travessão) é o formato real usado em
        # data/txt/gen_n2_010.txt — TITULO_CORPO exigia a linha ser só a
        # palavra do título, sem numeração.
        texto = "Interessado: FULANO\n\nI — RELATÓRIO\n\nConforme REsp 1.741.784/PR."
        ctx = analisar_contexto(texto)
        self.assertEqual(ctx.corpo_inicio, texto.index("I — RELATÓRIO"))

    def test_rotulo_com_ruido_de_ocr_e_reconhecido_pelo_fallback_generico(self):
        # "Apelãdo:" (ruído de OCR a->ã em "Apelado") é o caso real de
        # data/txt/gen_n2_013.txt — fora da enumeração de ROTULO, mas
        # reconhecido pelo fallback estrutural porque já há um campo antes.
        texto = "Processo 123456\nApelãdo: Ministério Público\n\nACÓRDÃO\nConforme REsp 1.741.784/PR."
        ctx = analisar_contexto(texto)
        rotulos = {c.rotulo for c in ctx.campos}
        self.assertIn("apelãdo", rotulos)

    def test_rotulos_de_parte_fora_da_enumeracao_sao_filtrados(self):
        # Vocabulário plausível nos 60% de documentos ocultos, que não está
        # na enumeração fixa de ROTULO — cobertos pelo fallback estrutural.
        for rotulo in ("Embargante", "Embargado", "Exequente", "Executado",
                       "Denunciado", "Litisconsorte", "Terceiro Interessado"):
            with self.subTest(rotulo=rotulo):
                texto = f"Processo 123456\n{rotulo}: FULANO DE TAL\n\nACÓRDÃO\nConforme REsp 1.741.784/PR."
                citacao = "REsp 1.741.784/PR"
                inicio = texto.index(citacao)
                item = {"inicio": inicio, "fim": inicio + len(citacao), "trecho": citacao}
                self.assertEqual(filtrar_distratores([item], texto), [item])
                nome_campo = texto.index(rotulo)
                campo = {"inicio": nome_campo, "fim": nome_campo + len(rotulo), "trecho": rotulo}
                self.assertTrue(e_distrator(campo, texto))

    def test_fallback_generico_nao_ativa_sem_campo_reconhecido_antes(self):
        # Sem nenhum campo de ROTULO reconhecido antes, um "Palavra: citação"
        # solto não deve ser tratado como cabeçalho (ex.: "Antes: REsp .../PR"
        # é só uma citação com prefixo qualquer, não um documento real).
        texto = "Antes: REsp 1.741.784/PR."
        citacao = "REsp 1.741.784/PR"
        inicio = texto.index(citacao)
        item = {"inicio": inicio, "fim": inicio + len(citacao), "trecho": citacao}
        self.assertEqual(filtrar_distratores([item], texto), [item])

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

    def test_citacao_na_mesma_linha_de_um_rotulo_qualquer_e_preservada(self):
        # Antes, um rótulo fora de processo/autos/valor tomava a linha
        # inteira como campo (fim = offset + len(linha)) e engolia junto
        # qualquer citação real na mesma linha — falso negativo silencioso.
        for texto, citacao in (
            ("Relator: Fulano de Tal, que também citou o REsp 1.741.784/PR em voto anterior.", "REsp 1.741.784/PR"),
            ("Ementa: agravo interno. Precedente firmado no REsp 1.741.784/PR. Recurso desprovido.", "REsp 1.741.784/PR"),
        ):
            with self.subTest(texto=texto):
                inicio = texto.index(citacao)
                item = {"inicio": inicio, "fim": inicio + len(citacao), "trecho": citacao}
                self.assertIsNone(motivo_distrator(item, texto))

    def test_campos_adjacentes_sem_citacao_continuam_fundidos_mesmo_com_numero(self):
        # Regressão da correção acima: "Memorial nº 255/2021" tem um número
        # que não é citação nenhuma. A truncatura desse campo não pode
        # quebrar a fusão com os campos anteriores ("Processo"/"Assistido")
        # e deixar esse trecho de cabeçalho sobreviver como candidato.
        texto = (
            "Processo nº 1292746-27.2020.7.13.1173\n"
            "Assistido: TRANSPORTES MARAJÁ EIRELI\n"
            "Memorial nº 255/2021\n\nMEMORIAL\n\nConforme REsp 1.741.784/PR."
        )
        trecho = texto[texto.index("Processo"):texto.index("255/2021") + len("255/2021")]
        item = {"inicio": texto.index(trecho), "fim": texto.index(trecho) + len(trecho), "trecho": trecho}
        self.assertIsNotNone(motivo_distrator(item, texto))

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
