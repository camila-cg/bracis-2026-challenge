"""ETAPA 5 — Resolução: (rota, trecho) -> classe + id_canonico + caminho."""

from __future__ import annotations

import sqlite3

import eda_utils as u
import indice_utils as ind
from modelos import CitacaoNormalizada, ResultadoResolucao
from steps.normalizar_citacao import normalizar_citacao
from steps.rotear_citacao import ROTA_ACORDAO, ROTA_DESCRITIVA, ROTA_LEI, ROTA_SUMULA, ROTA_TEMA


class Resolvedor:
    """Índices da base canônica e uma regra de resolução por rota."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.acordaos, _ = ind.construir_indice_acordaos(con)
        self.normativos, _ = ind.construir_indice_normativos(con)
        self.acordaos_por_numero = ind.indice_por_numero(self.acordaos)
        self.metadados_por_id = {
            registro["id"]: (tribunal, registro["classe"], registro["uf"])
            for (tribunal, _classe, _numero, _origem), registros in self.acordaos.items()
            for registro in registros
        }

    def resolver_citacao(self, citacao: CitacaoNormalizada) -> ResultadoResolucao:
        """Decide a classificação a partir dos atributos extraídos."""
        if citacao.familia == ROTA_LEI:
            if not citacao.artigo:
                # Sem artigo a citação não individualiza um dispositivo da
                # base, esteja a lei no catálogo ou não: é evasiva, não
                # alucinação. Só um número de lei zerado é inválido por si.
                if citacao.numero_lei and not citacao.numero_lei.strip("0"):
                    return ResultadoResolucao(
                        "inventada", None, "lei_inventada", cobertura="fora_base",
                        motivo="numero_zero",
                    )
                return ResultadoResolucao("incompleta", None, "lei_sem_artigo", cobertura="nao_consultada", motivo="artigo_ausente")
            if citacao.identidade_legal_valida is False:
                return ResultadoResolucao(
                    "inventada", None, "lei_inventada", cobertura="fora_base",
                    motivo="lei_numerada_sem_mapeamento",
                )
            if not citacao.diploma and citacao.lei_nomeada and citacao.contexto_legal_forte:
                return ResultadoResolucao(
                    "inventada", None, "lei_inventada", cobertura="fora_base",
                    motivo="diploma_nomeado_sem_mapeamento",
                )
            if not citacao.diploma:
                if citacao.lei_numerada:
                    # Convenção do desafio: lei numerada fora do catálogo congelado.
                    return ResultadoResolucao("inventada", None, "lei_inventada", cobertura="fora_base", motivo="lei_numerada_sem_mapeamento")
                return ResultadoResolucao("incompleta", None, "lei_sem_estatuto", cobertura="nao_consultada", motivo="diploma_ausente")
            alvo = self.normativos.get(f"lei|{citacao.diploma}|{citacao.artigo}")
            # Em ``arts. 5º e 7º`` o id é o do primeiro artigo, mas um artigo
            # coordenado inexistente é identificador explícito inválido.
            if alvo and any(
                not self.normativos.get(f"lei|{citacao.diploma}|{artigo}")
                for artigo in citacao.artigos_coordenados
            ):
                return ResultadoResolucao(
                    "inventada", None, "lei_inventada", motivo="artigo_coordenado_inexistente",
                )
            return self._resultado_normativo(alvo, "lei")

        if citacao.qualificador_invalido and citacao.numero:
            caminho = {
                ROTA_SUMULA: "sumula_inventada",
                ROTA_TEMA: "tema",
            }.get(citacao.familia, "acordao_zero")
            return ResultadoResolucao(
                "inventada", None, caminho, cobertura="fora_base",
                motivo="qualificador_invalido",
            )

        if citacao.familia == ROTA_SUMULA:
            if citacao.numero and not citacao.numero.strip("0"):
                return ResultadoResolucao("inventada", None, "sumula_inventada", motivo="numero_zero")
            if not citacao.numero:
                return ResultadoResolucao("incompleta", None, "sumula_sem_numero", cobertura="nao_consultada", motivo="numero_ausente")
            tribunal = "STF" if citacao.vinculante else citacao.tribunal
            if citacao.enunciado and not tribunal:
                # Enunciados de órgãos que não são tribunais (Fórum Nacional da Infância,
                # Jornadas de Direito Civil) não estão no catálogo: identificador explícito
                # sem mapeamento é inventado, como a lei numerada fora do catálogo.
                return ResultadoResolucao(
                    "inventada", None, "sumula_inventada", cobertura="fora_base",
                    motivo="enunciado_sem_mapeamento",
                )
            if not tribunal:
                return ResultadoResolucao("incompleta", None, "sumula_sem_tribunal", cobertura="nao_consultada", motivo="tribunal_ausente")
            numero = f"VINCULANTE-{citacao.numero}" if citacao.vinculante else citacao.numero
            alvo = self.normativos.get(f"sumula|{tribunal}|{numero}")
            return self._resultado_normativo(alvo, "sumula")

        if citacao.familia == ROTA_TEMA:
            if citacao.numero and not citacao.numero.strip("0"):
                return ResultadoResolucao("inventada", None, "tema", cobertura="fora_base", motivo="numero_zero")
            if not citacao.numero or not citacao.identificador_suficiente:
                return ResultadoResolucao("incompleta", None, "tema", cobertura="nao_consultada", motivo="numero_ausente")
            return ResultadoResolucao("inventada", None, "tema", cobertura="fora_base", motivo="tema_nao_indexado")

        if citacao.identificador_adulterado:
            return ResultadoResolucao(
                "inventada", None, "acordao_zero", cobertura="fora_base",
                motivo="cnj_malformado",
            )

        if citacao.familia == ROTA_DESCRITIVA or not citacao.numero:
            return ResultadoResolucao("incompleta", None, "descritiva", cobertura="nao_consultada", motivo="numero_ausente")

        candidatos = self._buscar_acordaos(citacao)
        caminho = "acordao_indice"
        if not candidatos:
            # O fallback recebe somente a identidade já selecionada. Consultar o
            # span inteiro poderia resolver por um ano ou outro número incidental.
            candidatos = set(u.resolver_por_numero(citacao.numero, con=self.con))
            candidatos, inconclusivos = self._filtrar_ids(citacao, candidatos)
            caminho = "acordao_fts"
            if inconclusivos:
                # O número aparece no cabeçalho, mas o parser não conseguiu
                # confirmar classe/UF explícitas. Escolher o único ID seria um
                # falso `real`; descartar seria afirmar que a citação é falsa
                # sem evidência. Mantemos todos como ambiguidade controlada.
                possiveis = tuple(sorted(candidatos | inconclusivos))
                return ResultadoResolucao(
                    "incompleta", None, "acordao_ambigua", possiveis,
                    motivo="metadados_nao_verificaveis",
                )
        if len(candidatos) == 1:
            return ResultadoResolucao("real", next(iter(candidatos)), caminho, tuple(sorted(candidatos)))
        if not candidatos:
            return ResultadoResolucao("inventada", None, "acordao_zero")
        return ResultadoResolucao("incompleta", None, "acordao_ambigua", tuple(sorted(candidatos)), motivo="multiplos_registros")

    @staticmethod
    def _resultado_normativo(alvos: list[dict] | None, familia: str) -> ResultadoResolucao:
        ids = tuple(sorted({item["id"] for item in alvos or []}))
        if len(ids) == 1:
            return ResultadoResolucao("real", ids[0], f"{familia}_real", ids)
        if ids:
            return ResultadoResolucao("incompleta", None, f"{familia}_ambigua", ids, motivo="multiplos_registros")
        return ResultadoResolucao("inventada", None, f"{familia}_inventada")

    def _buscar_acordaos(self, citacao: CitacaoNormalizada) -> set[int]:
        registros = self.acordaos_por_numero.get(citacao.numero or "", [])
        compativeis = []
        for registro in registros:
            if self._compativel(citacao, self.metadados_por_id[registro["id"]]):
                compativeis.append(registro)
        ids = {registro["id"] for registro in compativeis}
        if len(ids) <= 1:
            return ids
        # A classe-balde de _compativel ignora o estágio recursal (AgInt/AgRg/
        # EDcl/embargos de divergência) — é o que mais distingue acórdãos que
        # colidem por número dentro do mesmo tribunal/UF/classe. Tentamos
        # desempatar por isso primeiro, mas SÓ adotamos se sobrar exatamente
        # 1: o campo "classe" do cabeçalho às vezes registra apenas a classe
        # de base (ex. "RECLAMAÇÃO") mesmo quando o acórdão é um "AgInt"
        # sobre ela — se aplicássemos isso como filtro rígido, um caso desses
        # zerava os candidatos e uma citação real virava "inventada". Por
        # isso é desempate (nunca reduz a zero), não filtro.
        modificadores_citacao = ind._modificadores_citados(citacao.trecho)
        if modificadores_citacao:
            por_modificador = {
                registro["id"] for registro in compativeis
                if modificadores_citacao == ind._modificadores_citados(registro["classe"])
            }
            if len(por_modificador) == 1:
                return por_modificador
        # O parser do TSE às vezes deixa o preâmbulo "TRIBUNAL SUPERIOR
        # ELEITORAL ACÓRDÃO" em apenas um dos registros duplicados. Comparar a
        # classe bruta favoreceria artificialmente o registro sem esse ruído e
        # transformaria uma ambiguidade real em um ID escolhido às cegas.
        trecho_norm = ind._normalizar_para_comparar(citacao.normalizado)
        batidas = {
            registro["id"] for registro in compativeis
            if ind._classe_para_desempate(registro["classe"])
            and ind._classe_para_desempate(registro["classe"]) in trecho_norm
        }
        return batidas if len(batidas) == 1 else ids

    @staticmethod
    def _compativel(citacao: CitacaoNormalizada, metadados: tuple[str, str, str | None]) -> bool:
        tribunal, classe, uf = metadados
        classe_registro = ind._classe_citada(classe)
        return (
            (not citacao.tribunal or citacao.tribunal == tribunal)
            and (not citacao.uf or not uf or citacao.uf == uf)
            and (not citacao.classe or not classe_registro
                 or ind._classe_compativel(citacao.classe, classe_registro))
        )

    def _filtrar_ids(self, citacao: CitacaoNormalizada, ids: set[int]) -> tuple[set[int], set[int]]:
        """Separa candidatos compatíveis dos que não têm metadados verificáveis.

        O segundo conjunto não é tratado como compatível nem descartado: ele
        impede que o fallback transforme um número real com classe/UF falsa em
        `real`, sem chamar uma citação correta de `inventada` só porque o OCR do
        cabeçalho impediu o parser de confirmar seus qualificadores.
        """
        ausentes = ids - self.metadados_por_id.keys()
        if ausentes:
            placeholders = ",".join("?" for _ in ausentes)
            for id_canonico, tribunal, texto in self.con.execute(
                f"SELECT id, tribunal, texto FROM documentos WHERE id IN ({placeholders})", tuple(ausentes)
            ):
                campos = ind.parsear_cabecalho(tribunal, texto)
                self.metadados_por_id[id_canonico] = (
                    tribunal, campos["classe"] if campos else "",
                    ind.normalizar_uf(campos["uf"]) if campos else None,
                )
        compativeis = set()
        inconclusivos = set()
        for id_canonico in ids:
            metadados = self.metadados_por_id.get(id_canonico)
            if metadados is None:
                inconclusivos.add(id_canonico)
                continue
            tribunal, classe, uf = metadados
            if citacao.tribunal and citacao.tribunal != tribunal:
                continue
            classe_registro = ind._classe_citada(classe)
            if (citacao.classe and not classe_registro) or (citacao.uf and not uf):
                inconclusivos.add(id_canonico)
                continue
            if self._compativel(citacao, metadados):
                compativeis.add(id_canonico)
        return compativeis, inconclusivos

    def lei(self, trecho: str) -> tuple[str, int | None, str]:
        c = normalizar_citacao({"trecho": trecho, "inicio": 0, "fim": len(trecho)}, ROTA_LEI)
        r = self.resolver_citacao(c)
        return r.classe, r.id_canonico, r.caminho

    def sumula(self, trecho: str) -> tuple[str, int | None, str]:
        c = normalizar_citacao({"trecho": trecho, "inicio": 0, "fim": len(trecho)}, ROTA_SUMULA)
        r = self.resolver_citacao(c)
        return r.classe, r.id_canonico, r.caminho

    def tema(self, trecho: str) -> tuple[str, int | None, str]:
        c = normalizar_citacao({"trecho": trecho, "inicio": 0, "fim": len(trecho)}, ROTA_TEMA)
        r = self.resolver_citacao(c)
        return r.classe, r.id_canonico, r.caminho

    def descritiva(self, trecho: str) -> tuple[str, int | None, str]:
        c = normalizar_citacao({"trecho": trecho, "inicio": 0, "fim": len(trecho)}, ROTA_DESCRITIVA)
        r = self.resolver_citacao(c)
        return r.classe, r.id_canonico, r.caminho

    def acordao(self, trecho: str) -> tuple[str, int | None, str]:
        c = normalizar_citacao({"trecho": trecho, "inicio": 0, "fim": len(trecho)}, ROTA_ACORDAO, ja_normalizado=True)
        r = self.resolver_citacao(c)
        return r.classe, r.id_canonico, r.caminho


def resolver(rota: str, trecho: str, resolvedor: Resolvedor) -> tuple[str, int | None, str]:
    regra = {
        ROTA_LEI: resolvedor.lei,
        ROTA_SUMULA: resolvedor.sumula,
        ROTA_TEMA: resolvedor.tema,
        ROTA_DESCRITIVA: resolvedor.descritiva,
        ROTA_ACORDAO: resolvedor.acordao,
    }[rota]
    return regra(trecho)
