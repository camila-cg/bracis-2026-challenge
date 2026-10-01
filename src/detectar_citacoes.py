
""""
marcador (de vocabulário) -> termo do vocabulário que será procurado
marcador principal   -> ocorrência desse termo no texto
indício (complementar)  ->  informação próxima ao marcador principal, como número, tribunal, UF, relatoria, ano ou nome próprio (espera-se wue de pessoa)
registro -> guarda indícios e marcadores
citação -> grupo formado por um marcador principal e seus indícios complementares.
"""

import argparse
import csv
import json
import re
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from configuracao import carregar_config


caminho = Path(__file__).resolve().parent.parent / "data"
goldenset = caminho / "goldenset_offsets.csv"
documentos = caminho / "txt"


# Vocabulário principal, baseado no goldenset.
#BoW nas citações do gabarito + complementação com IA
_TRIBUNAIS = carregar_config("tribunais")
_CLASSES = carregar_config("classes_processuais")
_DIPLOMAS = carregar_config("diplomas")
_PIPELINE = carregar_config("pipeline")
MARCADORES_VOCABULARIO = {
    "jurisprudencia": _CLASSES["marcadores_jurisprudencia"],
    "lei": _DIPLOMAS["marcadores_lei"],}
MAX_TOKENS_MARCADOR = max(
    len(marcador.split())
    for marcadores in MARCADORES_VOCABULARIO.values()
    for marcador in marcadores
)
TRIBUNAIS = set(_TRIBUNAIS["siglas"])
UFS = set(_TRIBUNAIS["ufs"])
ORGAOS_JULGADORES = _TRIBUNAIS.get("orgaos_julgadores", [])
TERMOS_RELATORIA = set(_DIPLOMAS["termos_relatoria"])
COMPOSICAO_NOME = set(_DIPLOMAS["composicao_nome"])
MARCADORES_DESCRITIVOS = set(_CLASSES["marcadores_descritivos"])
MARCADORES_NUMERADOS = set(_CLASSES["marcadores_numerados"])
MARCADORES_LEGAIS_NUMERADOS = set(_DIPLOMAS["marcadores_numerados"])
ORGAOS_EMISSORES_RESOLUCAO = (
    TRIBUNAIS | set(_DIPLOMAS.get("orgaos_emissores_resolucao", []))
)
MARCADORES_ESTRUTURAIS_LEI = {"art", "artigo", "arts", "§"}
MARCADORES_NOVO_DIPLOMA = (
    set(_DIPLOMAS["marcadores_lei"]) - MARCADORES_ESTRUTURAIS_LEI
)
SIGLAS = set(_CLASSES["siglas"])
SIGLAS_CLASSES_AMBIGUAS = _CLASSES.get("siglas_classes_ambiguas", {})
MARCADORES_EXATOS = set(_CLASSES.get("marcadores_exatos", []))
PADROES_CLASSES_CITADAS = tuple(
    re.compile(item["padrao"], re.I)
    for item in _CLASSES["classes_citadas"]
)
IOU_MIN = 0.5
COBERTURA_EXTRA_MIN = 0.9

#para evitar que confuda com marcadores principais
TERMOS_PRIORITARIOS_INDICIOS = (TERMOS_RELATORIA | TRIBUNAIS)

PADRAO_DATA = re.compile(
    r"(?P<dia>\d{1,2})[./-]"
    r"(?P<mes>\d{1,2})[./-]"
    r"(?P<ano>\d{2}|\d{4})")

TERMOS_DATA_JULGAMENTO = { "julgamento", "julgado", "julgada", "sessao",}

TERMOS_DATA_PUBLICACAO = {"dje","dj", "publicado", "publicada", "publicacao",}


def verifica_data_valida(texto):
    match = PADRAO_DATA.fullmatch(texto)

    if match is None:
        return False

    dia = int(match.group("dia"))
    mes = int(match.group("mes"))
    ano = int(match.group("ano"))

    if ano < 100:
        ano += 2000

    try:
        date(ano, mes, dia)
    except ValueError:
        return False

    return True


def classificar_contexto_data(tokens, indice_data):
    inicio_contexto = max(0, indice_data - 3)

    termos_anteriores = {token["normalizado"] for token in tokens[inicio_contexto:indice_data]}
    token_anterior = (tokens[indice_data - 1]["normalizado"] if indice_data > 0 else None)
    if token_anterior == "j":
        return "data_julgamento"

    if termos_anteriores & TERMOS_DATA_JULGAMENTO:
        return "data_julgamento"

    if termos_anteriores & TERMOS_DATA_PUBLICACAO:
        return "data_publicacao"

    return "data"

def remover_acento(caractere):
    # Separa a letra do acento
    decomposto = unicodedata.normalize("NFD", caractere)

    base = ""
    # Remove o acento
    for parte in decomposto:
        if not unicodedata.combining(parte):
            base += parte
    return base.lower()

#Cria texto normalizado com o mesmo tamanho do texto original, preservando spans
def criar_texto_normalizado(texto_original):
    texto = texto_original.replace("\\n", "  ")
    caracteres = []

    for indice, caractere in enumerate(texto):
        anterior = texto[indice - 1] if indice > 0 else ""
        proximo = texto[indice + 1] if indice + 1 < len(texto) else ""

        #cobre \t e \n também
        if caractere.isspace():
            caracteres.append(" ")
            continue
        normalizado = remover_acento(caractere)
        if normalizado.isalnum() or normalizado in {"º", "°", "§"}:
            caracteres.append(normalizado)
            continue
        if caractere == "." and anterior.isdigit() and proximo.isdigit():
            caracteres.append(".")
            continue
        if caractere in {"/", "-"} and anterior.isalnum() and proximo.isalnum():
            caracteres.append(caractere)
            continue

        caracteres.append(" ")

    texto_normalizado = "".join(caracteres)
    if len(texto_normalizado) != len(texto_original):
        raise ValueError("A normalização alterou o tamanho do texto.")
    return texto_normalizado

#crie tokens com a forma original e normalizada, guardando os spans
def criar_tokens(texto_original):
    texto_normalizado = criar_texto_normalizado(texto_original)
    tokens = []

    for match in re.finditer(r"\S+", texto_normalizado):
        normalizado = match.group()
        tokens.append({"original": texto_original[match.start():match.end()],
            "normalizado": normalizado,
            "inicio": match.start(),
            "fim": match.end(),})
    return tokens

#para verificar acento no texto original
def verifica_acento(texto):
    texto_decomposto = unicodedata.normalize("NFD", texto)
    for caractere in texto_decomposto:
        if unicodedata.combining(caractere):
            return True
    return False


def criar_registro(tipo_registro, inicio_token, fim_token, tokens, pontos, **dados):
    registro = {
        "tipo_registro": tipo_registro,
        "inicio_token": inicio_token,
        "fim_token": fim_token,
        "inicio": tokens[inicio_token]["inicio"],
        "fim": tokens[fim_token]["fim"],
        "pontos": pontos,}
    registro.update(dados)
    return registro


#Calcula de 0 a 1 o quanto duas strings são parecidas, considerando caracteres em comum na mesma ordem, mesmo que estejam deslocados de posição
#Similaridade = 2*M / (len(a) + len(b)), onde M é a soma dos caracteres de todos os blocos correspondentes
def similaridade(a, b):
    return SequenceMatcher(None, a, b).ratio()

#procura marcador mais parecido no vocabulário
def encontrar_marcador_parecido(expressao, opcoes, threshold=0.80):
    melhor_opcao = None
    melhor_score = 0
    for opcao in opcoes:
        #Siglas muito curtas geram falsos positivos com similaridade aproximada
        expressao = expressao.rstrip(".")
        #se o marcador do vocabulário for muito curto, ignora se não for exatamente igual à expressão
        if (len(opcao) <= 3 or len(expressao) <= 3) and expressao!=opcao:
            continue
        partes_expressao = expressao.split()
        partes_opcao = opcao.split()
        if len(partes_opcao) > 1:
            parte_muito_diferente = False

            for parte_expressao, parte_opcao in zip(partes_expressao, partes_opcao):
                score_parte = similaridade(parte_expressao, parte_opcao)
                if score_parte <0.50:
                    parte_muito_diferente = True
                    break
            if parte_muito_diferente:
                continue
        score = similaridade(expressao, opcao)
        if score > melhor_score:
            melhor_score = score
            melhor_opcao = opcao
    if melhor_score >= threshold:
        return melhor_opcao, melhor_score
    return None, melhor_score

def tem_numero_logo_depois(indice, tokens):
    proximos = tokens[indice + 1: indice + 3]
    if not proximos:
        return False

    if re.search(r"\d", proximos[0]["normalizado"]):
        return True

    indicadores_de_numero = {"n", "nº", "n°", "no", "numero"}
    return (proximos[0]["normalizado"] in indicadores_de_numero
        and len(proximos) >= 2
        and re.search(r"\d", proximos[1]["normalizado"]) is not None)


def tem_numero_no_token_seguinte(indice, tokens):
    """Verdadeiro apenas quando o próximo token já contém um algarismo."""
    return (
        indice + 1 < len(tokens)
        and re.search(r"\d", tokens[indice + 1]["normalizado"]) is not None
    )


_INDICADORES_NUMERO_TEMA = ((), ("n",), ("nº",), ("n°",), ("no",), ("numero",))
_QUALIFICADORES_TEMA = (
    (),
    ("repetitivo",),
    ("de", "repercussao", "geral"),
    ("da", "repercussao", "geral"),
)


def tema_tem_identificador_juridico(indice, tokens):
    """Exige número após conectores próprios de um Tema jurisprudencial.

    A proximidade genérica não basta: em ``Tema em debate: 27``, por
    exemplo, 27 numera um assunto, não um precedente. Os padrões aceitos
    preservam as formas usuais ``Tema 27``, ``Tema nº 27``,
    ``Tema repetitivo 27`` e ``Tema de repercussão geral nº 27``.
    """
    posteriores = [token["normalizado"] for token in tokens[indice + 1:]]
    for qualificador in _QUALIFICADORES_TEMA:
        for indicador in _INDICADORES_NUMERO_TEMA:
            prefixo = qualificador + indicador
            if posteriores[:len(prefixo)] != list(prefixo):
                continue
            posicao_numero = len(prefixo)
            if (
                posicao_numero < len(posteriores)
                and re.search(r"\d", posteriores[posicao_numero]) is not None
            ):
                return True
    return False


_MARCADORES_SUMULA = {"sumula", "sum", "sumula vinculante"}
_INDICADORES_NUMERO_SUMULA = {"n", "nº", "n°", "no", "numero"}
_CONECTORES_TRIBUNAL_SUMULA = {"do", "da", "de"}


def _separador_continua_sumula(
    texto_original,
    token_anterior,
    token_seguinte,
    *,
    ponto_obrigatorio=False,
    ponto_opcional=False,
):
    """Valida uma transição da estrutura local de uma Súmula."""
    separador = texto_original[token_anterior["fim"]:token_seguinte["inicio"]]
    if separador.startswith("."):
        if not (ponto_obrigatorio or ponto_opcional):
            return False
        separador = separador[1:]
    elif ponto_obrigatorio:
        return False
    return re.search(r"[.,;:!?]", separador) is None


def sumula_tem_identificador(indice, tokens, texto_original):
    """Exige número local, com tribunal e indicador numérico opcionais."""
    marcador = tokens[indice]
    proximo_indice = indice + 1
    if proximo_indice >= len(tokens):
        return False

    if not _separador_continua_sumula(
        texto_original,
        marcador,
        tokens[proximo_indice],
        ponto_obrigatorio=marcador["normalizado"] == "sum",
    ):
        return False

    if tokens[proximo_indice]["normalizado"] in _CONECTORES_TRIBUNAL_SUMULA:
        conector = tokens[proximo_indice]
        proximo_indice += 1
        if (
            proximo_indice >= len(tokens)
            or tokens[proximo_indice]["normalizado"] not in TRIBUNAIS
            or not _separador_continua_sumula(
                texto_original, conector, tokens[proximo_indice]
            )
        ):
            return False

    if tokens[proximo_indice]["normalizado"] in TRIBUNAIS:
        tribunal = tokens[proximo_indice]
        proximo_indice += 1
        if (
            proximo_indice >= len(tokens)
            or not _separador_continua_sumula(
                texto_original, tribunal, tokens[proximo_indice]
            )
        ):
            return False

    indicador = tokens[proximo_indice]["normalizado"]
    if indicador in _INDICADORES_NUMERO_SUMULA:
        token_indicador = tokens[proximo_indice]
        proximo_indice += 1
        if (
            proximo_indice >= len(tokens)
            or not _separador_continua_sumula(
                texto_original,
                token_indicador,
                tokens[proximo_indice],
                ponto_opcional=indicador == "n",
            )
        ):
            return False

    return re.search(r"\d", tokens[proximo_indice]["normalizado"]) is not None


_INDICADORES_NUMERO_RESOLUCAO = {"n", "nº", "n°", "no", "numero"}


def _token_e_orgao_emissor_resolucao(token):
    """Reconhece emissores conhecidos e outras siglas institucionais."""
    if token["normalizado"] in ORGAOS_EMISSORES_RESOLUCAO:
        return True
    return re.fullmatch(r"[A-Z]{2,8}(?:-\d+)?", token["original"]) is not None


def _separador_continua_resolucao(
    texto_original, token_anterior, token_seguinte, *, abreviacao=False
):
    """Impede que a estrutura da resolução atravesse pontuação."""
    separador = texto_original[token_anterior["fim"]:token_seguinte["inicio"]]
    if abreviacao:
        # O ponto de ``Res.`` pertence ao marcador, não encerra a citação.
        if not separador.startswith("."):
            return False
        separador = separador[1:]
    return re.search(r"[.,;:!?]", separador) is None


def resolucao_tem_identificador(indice, tokens, texto_original):
    """Valida ``Res./Resolução + emissor? + nº? + identificador``.

    Sem introdutor de número, exige um identificador com barra para não
    transformar um número solto em prosa em ato normativo.
    """
    marcador = tokens[indice]
    abreviacao = marcador["normalizado"] == "res"
    proximo_indice = indice + 1
    if proximo_indice >= len(tokens):
        return False

    if not _separador_continua_resolucao(
        texto_original,
        marcador,
        tokens[proximo_indice],
        abreviacao=abreviacao,
    ):
        return False

    if _token_e_orgao_emissor_resolucao(tokens[proximo_indice]):
        anterior = tokens[proximo_indice]
        proximo_indice += 1
        if (
            proximo_indice >= len(tokens)
            or not _separador_continua_resolucao(
                texto_original, anterior, tokens[proximo_indice]
            )
        ):
            return False

    tem_indicador = (
        tokens[proximo_indice]["normalizado"]
        in _INDICADORES_NUMERO_RESOLUCAO
    )
    if tem_indicador:
        anterior = tokens[proximo_indice]
        proximo_indice += 1
        if (
            proximo_indice >= len(tokens)
            or not _separador_continua_resolucao(
                texto_original, anterior, tokens[proximo_indice]
            )
        ):
            return False

    identificador = tokens[proximo_indice]["normalizado"]
    if re.fullmatch(r"[0-9o]+(?:[./-][0-9o]+)*", identificador) is None:
        return False
    if not re.search(r"\d", identificador):
        return False
    return tem_indicador or "/" in identificador

#siglas ambíguas nao viram marcadores principais
def procurar_classes_processuais_ambiguas(tokens):
    classes = []

    for indice, token in enumerate(tokens):
        sigla = token["normalizado"]
        regra = SIGLAS_CLASSES_AMBIGUAS.get(sigla)

        if regra is None or not tem_numero_logo_depois(indice, tokens):
            continue

        # A normalização remove acentos; sem esta guarda, "aí 123456" seria
        # indistinguível da classe processual "AI 123456".
        if verifica_acento(token["original"]):
            continue

        token_anterior = ( tokens[indice - 1]["normalizado"] if indice > 0 else None)
        if token_anterior in regra.get("tokens_anteriores_bloqueados", []):
            continue

        classes.append(criar_registro(
            "classe_processual",
            indice,
            indice,
            tokens,
            regra.get("pontos", 1),
            valor=sigla,
            classe=regra["classe"],
            categoria="jurisprudencia",
            aceita_numero_processual_sem_contexto=regra.get(
                "aceita_numero_processual_sem_contexto", False
            ),
        ))

    return classes



_OCR_DE_LEI = {"le1", "lel"}
_ATOS_QUE_EXIGEM_IDENTIFICADOR_LOCAL = {
    "lei", "lei complementar", "decreto", "decr",
}
_QUALIFICADORES_ATO = {
    "federal", "estadual", "municipal", "distrital", "nacional",
    "ordinaria", "organica", "presidencial", "fed", "est", "mun",
}


def ato_tem_identificador_local(indice, fim_indice, tokens, texto_original):
    """Confirma número junto do ato ou artigo explícito imediatamente antes.

    Evita ligar ``Lei estadual pertinente`` ou ``lei anterior`` a um número
    solto mais adiante. Qualificadores institucionais são aceitos; para
    ``decreto presidencial``, o indicador ``nº`` é obrigatório.
    """
    inicio_contexto = max(0, indice - 8)
    contexto_anterior = tokens[inicio_contexto:indice]
    tem_artigo = any(
        token["normalizado"] in _MARCADORES_ARTIGO
        for token in contexto_anterior
    )
    tem_numero = any(
        re.search(r"\d", token["normalizado"])
        for token in contexto_anterior
    )
    token_anterior = tokens[indice - 1]["normalizado"] if indice else ""
    tem_artigo_anterior = (
        tem_artigo
        and tem_numero
        and (
            token_anterior in {"da", "do", "de", "na", "no"}
            or re.search(r"\d", token_anterior) is not None
        )
    )
    if tem_artigo_anterior:
        return True

    atual = fim_indice + 1
    if atual >= len(tokens):
        return False

    separador = texto_original[tokens[fim_indice]["fim"]:tokens[atual]["inicio"]]
    if tokens[fim_indice]["normalizado"] == "decr" and separador.startswith("."):
        separador = separador[1:]
    if re.search(r"[.;:!?]", separador):
        return False

    qualificador = None
    if (
        atual + 3 < len(tokens)
        and [token["normalizado"] for token in tokens[atual:atual + 3]]
        == ["do", "estado", "de"]
        and tokens[atual + 3]["normalizado"] in UFS
    ):
        atual += 4
        if atual >= len(tokens):
            return False

    if tokens[atual]["normalizado"] in _QUALIFICADORES_ATO:
        qualificador = tokens[atual]["normalizado"]
        atual += 1
        if atual >= len(tokens):
            return False

    tem_indicador = tokens[atual]["normalizado"] in _INDICADORES_NUMERO_PROCESSUAL
    if tem_indicador:
        atual += 1
        if atual >= len(tokens):
            return False

    if qualificador == "presidencial" and not tem_indicador:
        return False
    return re.search(r"\d", tokens[atual]["normalizado"]) is not None


def acordao_esta_em_formula_de_relatoria(indice, tokens, texto_original):
    """Reconhece ``rel. p/ o acórdão`` como parte da citação anterior."""
    prefixo = texto_original[
        max(0, tokens[indice]["inicio"] - 40):tokens[indice]["inicio"]
    ]
    return re.search(
        r"\brel(?:\.|ator(?:a)?)?\s+(?:p\s*/?|para)\s+o\s+$",
        prefixo,
        re.IGNORECASE,
    ) is not None


def procurar_marcadores_principais(tokens, texto_original):
    marcadores_principais = []
    intervalos_usados = set()

    for indice in range(len(tokens)):
        token_atual = tokens[indice]["normalizado"]
        if token_atual in TERMOS_PRIORITARIOS_INDICIOS:
            continue

        marcador_encontrado = None

        #primeiro verificar siglas compostas (tentar resolver o problema do AgR-REspe)
        partes = set(re.findall(r"[a-z]+", token_atual))
        tem_hifen = "-" in token_atual
        tem_sigla = False
        
        for parte in partes:
            if parte in SIGLAS:
                tem_sigla = True
                break
        if tem_hifen and tem_sigla:
            marcador_encontrado = ("jurisprudencia", token_atual, 1.0, indice)

        # "lei" é curto demais para a correspondência aproximada; o OCR que troca
        # o "i" por "1" ou "l" só vale quando um número vem logo depois.
        if (marcador_encontrado is None and token_atual in _OCR_DE_LEI
                and tem_numero_logo_depois(indice, tokens)):
            marcador_encontrado = ("lei", "lei", 0.9, indice)

         #agora somente procura no vocabulário se não for uma sigla composta
        if marcador_encontrado is None:
            for tamanho in range(MAX_TOKENS_MARCADOR, 0, -1):
                #fim exclusivo
                fim = indice + tamanho
                if fim > len(tokens):
                    continue

                expressao = " ".join(token["normalizado"] for token in tokens[indice:fim])

                for categoria, marcadores in MARCADORES_VOCABULARIO.items():
                    marcadores_mesmo_tamanho = [marcador for marcador in marcadores if len(marcador.split()) == tamanho]
                    #expressão exta
                    if expressao in marcadores_mesmo_tamanho:
                        marcador_encontrado = (categoria, expressao, 1.0, fim - 1)
                        break
                    #marcadores configurados como exatos não participam da correspondência aproximada.
                    marcadores_aproximados = [ marcador for marcador in marcadores_mesmo_tamanho if marcador not in MARCADORES_EXATOS]

                    marcador, similaridade = encontrar_marcador_parecido( expressao, marcadores_aproximados)
                    
                    if marcador is not None:
                        marcador_encontrado = (categoria, marcador, similaridade, fim - 1)
                        break

                if marcador_encontrado is not None:
                    break
        if marcador_encontrado is None:
            continue
        categoria, marcador, similaridade, fim_indice = marcador_encontrado

        # "SV" é uma abreviação curta e ambígua. Ela só vale como Súmula
        # Vinculante quando o token contém exatamente essas duas letras,
        # independentemente da capitalização, e o token seguinte já contém o
        # número. Não há correspondência aproximada nem correção de OCR.
        if marcador == "sv" and (
            fim_indice != indice
            or tokens[indice]["original"].upper() != "SV"
            or not tem_numero_no_token_seguinte(indice, tokens)
        ):
            continue

        # "Tema" também ocorre como substantivo comum. Um número apenas
        # próximo não o transforma em precedente; ele precisa aparecer logo
        # após o marcador ou após conectores jurídicos conhecidos.
        if marcador == "tema" and not tema_tem_identificador_juridico(
            fim_indice, tokens
        ):
            continue

        if marcador in _MARCADORES_SUMULA and not sumula_tem_identificador(
            fim_indice, tokens, texto_original
        ):
            continue

        if marcador in {"res", "resolucao"} and not resolucao_tem_identificador(
            fim_indice, tokens, texto_original
        ):
            continue

        if (
            marcador in _ATOS_QUE_EXIGEM_IDENTIFICADOR_LOCAL
            and not ato_tem_identificador_local(
                indice, fim_indice, tokens, texto_original
            )
        ):
            continue

        if marcador == "acordao" and acordao_esta_em_formula_de_relatoria(
            indice, tokens, texto_original
        ):
            continue

        intervalo = (indice, fim_indice)
        if intervalo in intervalos_usados:
            continue

        #Siglas pequenas precisam coincidir com o token original, sem acento
        if fim_indice == indice and marcador in SIGLAS:
            original = tokens[indice]["original"]
            if verifica_acento(original):
                continue

        pontos = 3 if similaridade == 1 else 2
        marcadores_principais.append(criar_registro("marcador_principal",indice,fim_indice,tokens,pontos,
        categoria=categoria, marcador=marcador,similaridade=round(similaridade, 3),))
        intervalos_usados.add(intervalo)

    return marcadores_principais


_INDICADORES_NUMERO_PROCESSUAL = {"n", "nº", "n°", "numero"}


def numero_curto_em_contexto_forte(indice, tokens):
    """Reconhece 1–2 dígitos só depois de classe conhecida e `nº` equivalente."""
    texto = tokens[indice]["normalizado"]
    curto = re.fullmatch(r"\d{1,2}(?:[/-]([a-z]{2}))?", texto)
    if curto is None or (curto.group(1) and curto.group(1) not in UFS):
        return False
    if indice == 0 or tokens[indice - 1]["normalizado"] not in _INDICADORES_NUMERO_PROCESSUAL:
        return False
    # Doze tokens cobrem as classes por extenso mais longas da configuração,
    # sem deixar uma classe distante qualificar qualquer número posterior.
    contexto = " ".join(
        token["normalizado"] for token in tokens[max(0, indice - 13):indice - 1]
    )
    return any(padrao.search(contexto) for padrao in PADROES_CLASSES_CITADAS)


def procurar_numeros(tokens):
    indicios = []

    for indice, token in enumerate(tokens):
        texto = token["normalizado"]
        if not re.search(r"\d", texto):
            continue

        #ano isolado
        if re.fullmatch(r"(?:19|20)\d{2}", texto):
            indicios.append(criar_registro("ano", indice, indice, tokens, 1, valor=texto))
            continue

        tem_formato_data = PADRAO_DATA.fullmatch(texto) is not None
        if tem_formato_data:
            tipo_data = classificar_contexto_data(tokens, indice)
            data_valida = verifica_data_valida(texto)
            tem_contexto_juridico = tipo_data != "data"

            #Data válida sempre registra.
            if data_valida:
                pontos = 1 if tem_contexto_juridico else 0

                indicios.append(criar_registro(tipo_data,indice,indice, tokens,pontos,valor=texto,data_valida=True,))
                continue

            #Data inválida com contexto jurídico preserva como possível erro de OCR, mas sem pontuar
            if tem_contexto_juridico:
                indicios.append(
                    criar_registro(tipo_data,indice,indice,tokens,0,valor=texto,data_valida=False,))
                continue

            #Data inválida sem contexto segue para a verificação de número processual.

        quantidade_digitos = len(re.findall(r"\d", texto))
        # Processos antigos podem ter somente 1–2 dígitos. O contexto forte os
        # distingue de quantidades soltas antes mesmo do roteamento.
        possui_formato_processual = (
            quantidade_digitos >= 4
            or bool(re.search(r"\d[./-]\d", texto))
            or bool(re.search(r"[a-z]+-.*\d", texto))
            or numero_curto_em_contexto_forte(indice, tokens)
        )

        tipo = "numero_processual" if possui_formato_processual else "numero_legal"
        pontos=1
        indicios.append(criar_registro(tipo, indice, indice, tokens, pontos, valor=texto))

    return indicios


_MARCADORES_ARTIGO = {"art", "artigo", "arts", "§"}
_QUALIFICADORES_INCISO = {"inc", "inciso", "incisos"}
_QUALIFICADORES_ALINEA = {"alinea", "alineas"}
_NUMERO_ROMANO = re.compile(r"[ivxlcdm]+")


def procurar_qualificadores_legais(texto_original, tokens):
    """Preserva incisos e alíneas explícitos no fim de uma citação legal."""
    registros = []
    for indice, token in enumerate(tokens):
        qualificador = token["normalizado"]
        if _NUMERO_ROMANO.fullmatch(qualificador):
            contexto = tokens[max(0, indice - 5):indice]
            tem_artigo = any(
                item["normalizado"] in _MARCADORES_ARTIGO for item in contexto
            )
            numero_anterior = (
                indice > 0
                and re.search(r"\d", tokens[indice - 1]["normalizado"])
            )
            if tem_artigo and numero_anterior:
                separador = texto_original[tokens[indice - 1]["fim"]:token["inicio"]]
                if re.search(r"[;!?]|\.\s*$", separador) is None:
                    fim_indice = indice
                    while fim_indice + 2 < len(tokens):
                        if tokens[fim_indice + 1]["normalizado"] != "e":
                            break
                        if _NUMERO_ROMANO.fullmatch(
                            tokens[fim_indice + 2]["normalizado"]
                        ) is None:
                            break
                        fim_indice += 2
                    registros.append(criar_registro(
                        "qualificador_legal", indice, fim_indice, tokens, 0,
                        valor=texto_original[token["inicio"]:tokens[fim_indice]["fim"]],
                    ))
            continue
        if qualificador not in (_QUALIFICADORES_INCISO | _QUALIFICADORES_ALINEA):
            continue

        contexto = tokens[max(0, indice - 5):indice]
        tem_artigo = any(
            item["normalizado"] in _MARCADORES_ARTIGO for item in contexto
        )
        tem_numero = any(
            re.search(r"\d", item["normalizado"]) is not None
            for item in contexto
        )
        if not (tem_artigo and tem_numero) or indice + 1 >= len(tokens):
            continue

        if indice > 0:
            separador = texto_original[tokens[indice - 1]["fim"]:token["inicio"]]
            if re.search(r"[;!?]|\.\s*$", separador):
                continue

        aceita_valor = (
            (qualificador in _QUALIFICADORES_INCISO
             and _NUMERO_ROMANO.fullmatch(tokens[indice + 1]["normalizado"]))
            or
            (qualificador in _QUALIFICADORES_ALINEA
             and re.fullmatch(r"[a-z]", tokens[indice + 1]["normalizado"]))
        )
        if not aceita_valor:
            continue

        fim_indice = indice + 1
        while fim_indice + 2 < len(tokens):
            conector = tokens[fim_indice + 1]["normalizado"]
            proximo = tokens[fim_indice + 2]["normalizado"]
            if conector != "e":
                break
            if qualificador in _QUALIFICADORES_INCISO:
                if _NUMERO_ROMANO.fullmatch(proximo) is None:
                    break
            elif re.fullmatch(r"[a-z]", proximo) is None:
                break
            fim_indice += 2

        registros.append(criar_registro(
            "qualificador_legal", indice, fim_indice, tokens, 0,
            valor=texto_original[token["inicio"]:tokens[fim_indice]["fim"]],
        ))
    return registros


def procurar_tribunais_e_ufs(tokens):
    indicios = []

    for indice, token in enumerate(tokens):
        texto = token["normalizado"]
        original = token["original"]

        tribunal_encontrado = None
        for tribunal in TRIBUNAIS:
            if texto == tribunal or texto.startswith(tribunal + "-"):
                tribunal_encontrado = tribunal
                break
        if tribunal_encontrado:
            indicios.append(criar_registro("tribunal", indice, indice, tokens, 1, valor=tribunal_encontrado.upper(),))

        uf_encontrada = None
        #Verifica se há algum número nos dois tokens anteriores
        perto_de_numero = False
        inicio_busca = max(0, indice - 2)

        for anterior in range(inicio_busca, indice):
            texto_anterior = tokens[anterior]["normalizado"]
            if re.search(r"\d", texto_anterior):
                perto_de_numero = True
                break
        #UF como token separdo
        if re.fullmatch(r"[A-Z]{2}", original):
            if texto in UFS:
                if perto_de_numero:
                    uf_encontrada = texto
        #UF aparece junto de número
        if uf_encontrada is None:
            match = re.search(r"(?:/|-)([a-z]{2})$", texto)
            if match:
                uf = match.group(1)
                if uf in UFS:
                    if re.search(r"\d", texto):
                        uf_encontrada = uf
        if uf_encontrada:
            indicios.append(criar_registro("uf", indice, indice, tokens, 1, valor=uf_encontrada.upper()))
    return indicios


def token_parece_nome(token):
    original = token["original"]
    normalizado = token["normalizado"]
    if normalizado in COMPOSICAO_NOME:
        return True
    if not original or re.search(r"\d", original):
        return False
    return original[0].isupper()


def procurar_relatorias_e_nomes(texto_original, tokens):
    indicios = []

    for indice, token in enumerate(tokens):
        if token["normalizado"] not in TERMOS_RELATORIA:
            continue
        else:
            indicios.append(criar_registro("relatoria", indice, indice, tokens, 1, valor=token["normalizado"],))

        inicio_nome = indice + 1
        while inicio_nome < len(tokens) and tokens[inicio_nome]["normalizado"] in (COMPOSICAO_NOME | {"min", "ministro", "ministra"}):
            inicio_nome += 1

        #aceita um possível erro de OCR de de, da ou do - existe no goldenset
        if (inicio_nome < len(tokens) and re.fullmatch(r"d[a-z]", tokens[inicio_nome]["normalizado"])):
            inicio_nome += 1

        partes = []
        fim_nome = None
        for proximo_indice in range(inicio_nome, min(inicio_nome + 7, len(tokens))):
            proximo = tokens[proximo_indice]
            if partes:
                separador = texto_original[tokens[proximo_indice - 1]["fim"]:proximo["inicio"]]
                if re.search(r"[.;:]|\n\s*\n", separador):
                    break
            if not token_parece_nome(proximo):
                break
            partes.append(proximo)
            fim_nome = proximo_indice

        partes_relevantes = []
        for parte in partes:
            if parte["normalizado"] not in COMPOSICAO_NOME:
                partes_relevantes.append(parte)
        if len(partes_relevantes) >= 2 and fim_nome is not None:
            indicios.append(criar_registro("nome_proprio", inicio_nome, fim_nome, tokens, 1, valor=" ".join(parte["original"] for parte in partes),))
    return indicios

def procurar_orgaos_julgadores(tokens):
    indicios = []

    #Expressões maiores são verificadas primeiro
    orgaos_ordenados = sorted(ORGAOS_JULGADORES, key=lambda orgao: len(orgao.split()), reverse=True,)

    for inicio in range(len(tokens)):
        for orgao in orgaos_ordenados:
            quantidade_tokens = len(orgao.split())
            fim_exclusivo = inicio + quantidade_tokens

            if fim_exclusivo > len(tokens):
                continue

            expressao = " ".join(token["normalizado"] for token in tokens[inicio:fim_exclusivo])

            if expressao != orgao:
                continue

            fim_indice = fim_exclusivo - 1

            indicios.append(criar_registro("orgao_julgador", inicio, fim_indice, tokens, 0, valor=orgao,))
            break

    return indicios

######### Essa parte debaixo foi totalmente feita por IA (e não toda foi revisada a fundo) para conseguir executar o código e testar o funcionamento das regras acima, ajustar algumas coisas... Ainda não fiz a parte oficial 
####### de juntar os registros identificados em citações, calcular a pontuação final, comparar com os documentos e dar um retorno
####### parece que está funcionando, mas tem que ser revisada e pelo que olhei não era exatamente como eu estava pensando, mas deve dar para aproveitar algumas coisas mais mecânicas e que não vale a pena perder muito tempo fazendo

#essa já foi revisada, está ok
def detectar_marcadores_e_indicios(texto_original, tokens):
    registros = []
    registros.extend(procurar_marcadores_principais(tokens, texto_original))
    registros.extend(procurar_classes_processuais_ambiguas(tokens))
    registros.extend(procurar_numeros(tokens))
    registros.extend(procurar_qualificadores_legais(texto_original, tokens))
    registros.extend(procurar_tribunais_e_ufs(tokens))
    registros.extend(procurar_orgaos_julgadores(tokens))
    registros.extend(procurar_relatorias_e_nomes(texto_original, tokens))
    registros_unicos = {}
    for registro in registros:
        chave = (registro["tipo_registro"],
            registro["inicio"],
            registro["fim"],
            registro.get("categoria"),)
        registros_unicos[chave] = registro

    return sorted(registros_unicos.values(),key=lambda item: (item["inicio_token"], item["fim_token"], item["tipo_registro"]),)

#essa já foi revisada, está ok
def limite_entre_citacoes(texto_original, registro_anterior, registro_atual):
    trecho = texto_original[registro_anterior["fim"]:registro_atual["inicio"]]
    if re.search(r"[;!?]", trecho):
        return True
    if (re.search(r"\.(?:\s|$)", trecho) and registro_atual["tipo_registro"] == "marcador_principal"):
        return True
    #novo paragrafo
    if "\n\n" in trecho:
        return True
    #vírgulas entre citações de jurisprudência
    if "," in trecho and registro_atual["tipo_registro"] == "marcador_principal":
        outro_marcador_jurisprudencia = (registro_atual.get("categoria") == "jurisprudencia")
        fim_citacao_anterior = registro_anterior["tipo_registro"] in {"numero_processual", "uf"}
        return outro_marcador_jurisprudencia and fim_citacao_anterior
    return False


_REGISTROS_NUMERICOS = {
    "numero_processual", "numero_legal", "ano", "data",
    "data_julgamento", "data_publicacao",
}
_FIM_DE_FRASE_APOS_REGISTRO = re.compile(
    r'^\s*[)\]}\"\'\u2019\u201d\u00bb]*\.\s+(?=[A-ZÀ-ÖØ-Þ])'
)


def numero_comeca_nova_frase(
    texto_original, grupo_atual, registro_anterior, registro_atual
):
    """Não anexa um número de outra frase a uma citação já identificada.

    A restrição só é ativada depois de marcador/classe + número e quando a frase
    seguinte começa por palavra em maiúscula. Assim, pontos de abreviações antes
    do identificador (``art. 5º``, ``n. 27``), evidências posteriores
    (``rel. Min.``) e números OCR quebrados em linha (``8O4.\n55O``) permanecem.
    """
    tem_inicio_juridico = any(
        item["tipo_registro"] in {"marcador_principal", "classe_processual"}
        for item in grupo_atual
    )
    tem_identificador = any(
        item["tipo_registro"] in {"numero_processual", "numero_legal"}
        for item in grupo_atual
    )
    anterior_encerra_identificador = registro_anterior["tipo_registro"] in (
        _REGISTROS_NUMERICOS | {"uf", "tribunal", "qualificador_legal"}
    )
    atual_e_numerico = registro_atual["tipo_registro"] in _REGISTROS_NUMERICOS
    separador = texto_original[
        registro_anterior["fim"]:registro_atual["inicio"]
    ]
    return (
        tem_inicio_juridico
        and tem_identificador
        and anterior_encerra_identificador
        and atual_e_numerico
        and _FIM_DE_FRASE_APOS_REGISTRO.match(separador) is not None
    )


def nova_fonte_coordenada(texto_original, grupo_atual, registro_atual):
    """Separa fontes autônomas ligadas por ``e`` ou ``bem como``.

    Marcadores estruturais de artigo permanecem no mesmo span. Para
    jurisprudência, uma nova classe/sigla depois de um identificador inicia
    outra citação; isso cobre pares como ``REsp ... e HC ...``.
    """
    if registro_atual.get("tipo_registro") not in {
        "marcador_principal", "classe_processual",
    }:
        return False

    categoria_atual = registro_atual.get("categoria")
    if (
        categoria_atual == "lei"
        and registro_atual.get("marcador") not in MARCADORES_NOVO_DIPLOMA
    ):
        return False

    categorias_anteriores = {
        item.get("categoria")
        for item in grupo_atual
        if item.get("tipo_registro") in {
            "marcador_principal", "classe_processual",
        }
    }
    tem_numero = any(
        item["tipo_registro"] in {"numero_processual", "numero_legal"}
        for item in grupo_atual
    )
    if categoria_atual not in categorias_anteriores or not tem_numero:
        return False

    separador = texto_original[grupo_atual[-1]["fim"]:registro_atual["inicio"]]
    return re.search(
        r"\b(?:e|bem\s+como)\s+(?:(?:o|a|os|as)\s+)?$",
        separador,
        re.IGNORECASE,
    ) is not None

#revisada, mas talvez tenhamos que mexer em algumas heurísticas futuramente
def agrupar_registros(texto_original, registros):
    grupos = []
    grupo_atual = []

    for registro in registros:
        if not grupo_atual:
            grupo_atual = [registro]
            continue

        anterior = grupo_atual[-1]
        distancia = registro["inicio_token"] - anterior["fim_token"] - 1
        #soma um para incluir as bordas
        tamanho_grupo = registro["fim_token"] - grupo_atual[0]["inicio_token"] + 1
        inicios_citacao_do_grupo = []

        for item in grupo_atual:
            if item["tipo_registro"] in {"marcador_principal", "classe_processual"}:
                inicios_citacao_do_grupo.append(item)

        grupo_ja_tem_numero = any(item["tipo_registro"] in {"numero_processual", "numero_legal"} for item in grupo_atual)

        novo_marcador_categoria_diferente = False

        registro_inicia_citacao = registro["tipo_registro"] in {"marcador_principal", "classe_processual"}

        if (registro_inicia_citacao and inicios_citacao_do_grupo and grupo_ja_tem_numero):
            ultimo_inicio_citacao = inicios_citacao_do_grupo[-1]

            categoria_anterior = ultimo_inicio_citacao.get("categoria")
            categoria_atual = registro.get("categoria")

            novo_marcador_categoria_diferente = (categoria_anterior is not None and categoria_atual is not None and categoria_anterior != categoria_atual)

        # Uma SV numerada é uma fonte autônoma. Se o grupo anterior já tem
        # número, ela inicia outra citação mesmo na mesma frase (por exemplo,
        # "Tema 1.076/STJ e SV 37").
        nova_sv_apos_citacao_numerada = (
            registro.get("tipo_registro") == "marcador_principal"
            and registro.get("marcador") == "sv"
            and grupo_ja_tem_numero
        )


        novo_marcador_distante = False
        if registro_inicia_citacao and inicios_citacao_do_grupo:
            ultimo_inicio_citacao = inicios_citacao_do_grupo[-1]
            distancia_marcadores = (registro["inicio_token"] - ultimo_inicio_citacao["fim_token"] - 1)
            #heurística para quando tem mais de uma citação na mesma frase/parágrafo. Parece boa
            novo_marcador_distante = (distancia_marcadores > _PIPELINE["distancia_marcadores_jurisprudencia"] and registro.get("categoria") == "jurisprudencia")

        numero_em_nova_frase = numero_comeca_nova_frase(
            texto_original, grupo_atual, anterior, registro
        )
        nova_fonte = nova_fonte_coordenada(
            texto_original, grupo_atual, registro
        )

        #outra heurística que talvez precise ser ajustada para generalização
        if (distancia <= _PIPELINE["distancia_maxima_grupo"] and tamanho_grupo <= _PIPELINE["tamanho_maximo_grupo"] and not novo_marcador_distante and not novo_marcador_categoria_diferente and not nova_sv_apos_citacao_numerada and not numero_em_nova_frase and not nova_fonte and not limite_entre_citacoes(texto_original, anterior, registro)):
            grupo_atual.append(registro)
        else:
            grupos.append(grupo_atual)
            grupo_atual = [registro]
    #salva o último grupo
    if grupo_atual:
        grupos.append(grupo_atual)
    return grupos


def parece_jurisprudencia_sem_marcador(grupo):
    tipos = {registro["tipo_registro"] for registro in grupo}

    tem_numero = "numero_processual" in tipos
    tem_tribunal = "tribunal" in tipos
    tem_uf = "uf" in tipos
    tem_relatoria = "relatoria" in tipos
    tem_orgao = "orgao_julgador" in tipos
    tem_data = bool(tipos & {"data_julgamento","data_publicacao",})
    tem_classe_processual = "classe_processual" in tipos

    # Algumas siglas exatas, embora ambíguas isoladamente, identificam a
    # fonte quando acompanhadas de um número com formato processual. A opção
    # é declarada por classe para não afrouxar MS e RO globalmente.
    tem_classe_autossuficiente = any(
        registro["tipo_registro"] == "classe_processual"
        and registro.get("aceita_numero_processual_sem_contexto", False)
        for registro in grupo
    )

    citacao_numerada_completa = (tem_numero and tem_uf and tem_tribunal)

    citacao_numerada_contextual = (tem_numero and (tem_tribunal or tem_uf) and (tem_relatoria or tem_orgao or tem_data))

    citacao_descritiva = (tem_tribunal and tem_relatoria and (tem_orgao or tem_data))

    citacao_com_classe_processual = (tem_classe_processual and tem_numero and (tem_tribunal or tem_uf or tem_relatoria or tem_orgao or tem_data))

    citacao_com_classe_e_identificador = (
        tem_classe_autossuficiente and tem_numero
    )

    return (
        citacao_numerada_completa
        or citacao_numerada_contextual
        or citacao_descritiva
        or citacao_com_classe_processual
        or citacao_com_classe_e_identificador
    )


#define a categoria dos registros - ok
def escolher_categoria(grupo):
    marcadores_principais = [ registro for registro in grupo
        if registro["tipo_registro"] == "marcador_principal"]
    if not marcadores_principais:
        if parece_jurisprudencia_sem_marcador(grupo):
            return "jurisprudencia"
        return None

    marcadores_lei = {
        registro["marcador"] for registro in marcadores_principais
        if registro["categoria"] == "lei"}
    if marcadores_lei & MARCADORES_LEGAIS_NUMERADOS:
        return "lei"

    pontos_por_categoria = {"jurisprudencia": 0, "lei": 0}
    for registro in marcadores_principais:
        pontos_por_categoria[registro["categoria"]] += registro["pontos"]
    return max(pontos_por_categoria, key=pontos_por_categoria.get)

#verifica se um grupo tem registros suficientes para poder ser considerado citação - ok, mas teve que ser ajustada
def grupo_atende_regras(grupo, categoria):
    marcadores_principais = [
        registro for registro in grupo
        if registro["tipo_registro"] == "marcador_principal"
        and registro["categoria"] == categoria]
    marcadores = {
        registro["marcador"] for registro in marcadores_principais}
    tipos = {registro["tipo_registro"] for registro in grupo}

    if categoria == "lei":
        tem_marcador_legal_numerado = bool(marcadores & MARCADORES_LEGAIS_NUMERADOS)
        tem_numero = bool(tipos & {"numero_legal", "numero_processual"})

        return tem_marcador_legal_numerado and tem_numero

    tem_numero_processual = "numero_processual" in tipos
    tem_algum_numero = bool(tipos & {"numero_processual", "numero_legal"})
    fragmentos_numericos = sum(registro["tipo_registro"] == "numero_legal" for registro in grupo)
    quantidade_indicios_descritivos = 0

    if "tribunal" in tipos:
        quantidade_indicios_descritivos += 1
    if "ano" in tipos:
        quantidade_indicios_descritivos += 1
    if "relatoria" in tipos:
        quantidade_indicios_descritivos += 1
    if "nome_proprio" in tipos:
        quantidade_indicios_descritivos += 1

    tem_descricao_identificavel = (quantidade_indicios_descritivos >= 2)

    if marcadores & MARCADORES_NUMERADOS:
        return tem_algum_numero
    return (tem_numero_processual or fragmentos_numericos >= 2 or tem_descricao_identificavel)

#ajustada, pontua os grupos para depois saber se vamos considerar como citação
def pontuar_grupo(grupo, categoria):
    pontos = 0
    registros_contados = set()

    for registro in grupo:
        if registro["tipo_registro"] == "marcador_principal":
            if registro["categoria"] != categoria:
                continue
        elif categoria == "lei" and registro["tipo_registro"] in {"tribunal", "relatoria"}:
            continue

        chave = (
            registro["tipo_registro"],
            registro["inicio"],
            registro["fim"],
        )
        if chave in registros_contados:
            continue
        pontos += registro["pontos"]
        registros_contados.add(chave)

    return pontos

# revisada
def formar_candidatos(texto_original, tokens, grupos):
    candidatos = []

    for grupo in grupos:
        categoria = escolher_categoria(grupo)
        if categoria is None or not grupo_atende_regras(grupo, categoria):
            continue

        registros_categoria = [
            registro for registro in grupo
            if registro["tipo_registro"] != "marcador_principal"
            or registro["categoria"] == categoria]
        
        inicio = min(registro["inicio"] for registro in registros_categoria)
        fim = max(registro["fim"] for registro in registros_categoria)
        pontos = pontuar_grupo(registros_categoria, categoria)

        candidatos.append({
            "tipo": categoria,
            "inicio": inicio,
            "fim": fim,
            "trecho": texto_original[inicio:fim],
            "pontuacao": pontos,
            "registros": [
                {"tipo": registro["tipo_registro"],
                    "valor": registro.get("marcador", registro.get("valor")),
                    "pontos": registro["pontos"],}
                for registro in registros_categoria
            ],
        })

    return candidatos

#mede o quanto do o span detectado coincide com o goldenset
def calcular_sobreposicao(candidato, anotacao):
    inicio = max(candidato["inicio"], int(anotacao["inicio"]))
    fim = min(candidato["fim"], int(anotacao["fim"]))
    intersecao = max(0, fim - inicio)
    if intersecao == 0:
        return 0

    uniao = (max(candidato["fim"], int(anotacao["fim"])) - min(candidato["inicio"], int(anotacao["inicio"])))
    return intersecao / uniao


def calcular_cobertura_da_predicao(candidato, anotacao):
    """Fração da predição coberta pelo gold, como na regra oficial EXTRA."""
    largura = candidato["fim"] - candidato["inicio"]
    if largura <= 0:
        return 0
    inicio = max(candidato["inicio"], int(anotacao["inicio"]))
    fim = min(candidato["fim"], int(anotacao["fim"]))
    return max(0, fim - inicio) / largura

#funcao para verificar se acertou a detecção do span
def avaliar_documento(candidatos, anotacoes):
    # Replica o matching oficial: IoU >= 0,5 e escolha gulosa pela maior
    # sobreposição. O formato de submissão não tem o campo ``tipo``; por isso
    # ele não participa do casamento de spans.
    pares_duplicados = [
        (primeiro, segundo)
        for primeiro in range(len(candidatos))
        for segundo in range(primeiro + 1, len(candidatos))
        if calcular_sobreposicao(candidatos[primeiro], candidatos[segundo])
        >= IOU_MIN
    ]
    candidatos_em_duplicatas = {
        indice for par in pares_duplicados for indice in par
    }

    pares_possiveis = []
    for indice_anotacao, anotacao in enumerate(anotacoes):
        for indice_candidato, candidato in enumerate(candidatos):
            sobreposicao = calcular_sobreposicao(candidato, anotacao)
            if sobreposicao >= IOU_MIN:
                pares_possiveis.append(
                    (-sobreposicao, indice_anotacao, indice_candidato)
                )

    anotacoes_usadas = set()
    candidatos_usados = set()
    correspondencias = []
    for negativo_iou, indice_anotacao, indice_candidato in sorted(pares_possiveis):
        if indice_anotacao in anotacoes_usadas or indice_candidato in candidatos_usados:
            continue
        anotacoes_usadas.add(indice_anotacao)
        candidatos_usados.add(indice_candidato)
        anotacao = anotacoes[indice_anotacao]
        candidato = candidatos[indice_candidato]
        correspondencias.append({
            "citacao_id": anotacao["citacao_id"],
            "indice_candidato": indice_candidato,
            "sobreposicao": round(-negativo_iou, 3),
            "span_exato": (
                candidato["inicio"] == int(anotacao["inicio"])
                and candidato["fim"] == int(anotacao["fim"])
            ),
        })

    # O scorer ignora componentes extras quase contidos em um gold que já
    # casou. Isso evita penalizar dois spans da mesma citação, sem reutilizar
    # uma predição como verdadeiro positivo.
    indices_extras_ignorados = {
        indice_candidato
        for indice_candidato, candidato in enumerate(candidatos)
        if indice_candidato not in candidatos_usados
        # O scorer oficial rejeita a submissão antes do matching quando duas
        # predições têm IoU >= 0,5. No diagnóstico local, não perdoamos a
        # duplicata como EXTRA: ela permanece FP e a invalidade fica explícita.
        and indice_candidato not in candidatos_em_duplicatas
        and any(
            calcular_cobertura_da_predicao(candidato, anotacoes[indice_anotacao])
            >= COBERTURA_EXTRA_MIN
            for indice_anotacao in anotacoes_usadas
        )
    }

    verdadeiros_positivos = len(correspondencias)
    falsos_positivos = (
        len(candidatos) - verdadeiros_positivos - len(indices_extras_ignorados)
    )
    falsos_negativos = len(anotacoes) - verdadeiros_positivos
    precisao = (
        verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
        if verdadeiros_positivos + falsos_positivos else 0
    )
    cobertura = (
        verdadeiros_positivos / len(anotacoes) if anotacoes else 0
    )
    f1 = (
        2 * precisao * cobertura / (precisao + cobertura)
        if precisao + cobertura else 0
    )

    return {
        "verdadeiros_positivos": verdadeiros_positivos,
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
        "extras_ignorados": len(indices_extras_ignorados),
        "duplicatas_invalidas": len(pares_duplicados),
        "submissao_valida": not pares_duplicados,
        "spans_exatos": sum(item["span_exato"] for item in correspondencias),
        "precisao": round(precisao, 4),
        "cobertura": round(cobertura, 4),
        "f1": round(f1, 4),
        "indices_correspondentes": {
            item["indice_candidato"] for item in correspondencias
        },
        "indices_extras_ignorados": indices_extras_ignorados,
    }

#avalia o conjunto do goldenset
def avaliar_conjunto(candidatos_por_documento, anotacoes_por_documento, threshold):
    total_candidatos = 0
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0
    extras_ignorados = 0
    duplicatas_invalidas = 0
    spans_exatos = 0

    for documento_id, anotacoes in anotacoes_por_documento.items():
        candidatos = [
            candidato
            for candidato in candidatos_por_documento[documento_id]
            if candidato["pontuacao"] >= threshold
        ]
        avaliacao = avaliar_documento(candidatos, anotacoes)
        total_candidatos += len(candidatos)
        verdadeiros_positivos += avaliacao["verdadeiros_positivos"]
        falsos_positivos += avaliacao["falsos_positivos"]
        falsos_negativos += avaliacao["falsos_negativos"]
        extras_ignorados += avaliacao["extras_ignorados"]
        duplicatas_invalidas += avaliacao["duplicatas_invalidas"]
        spans_exatos += avaliacao["spans_exatos"]

    total_esperadas = verdadeiros_positivos + falsos_negativos
    total_avaliadas = verdadeiros_positivos + falsos_positivos
    precisao = verdadeiros_positivos / total_avaliadas if total_avaliadas else 0
    cobertura = verdadeiros_positivos / total_esperadas if total_esperadas else 0
    f1 = (
        2 * precisao * cobertura / (precisao + cobertura)
        if precisao + cobertura else 0
    )
    return {
        "threshold": threshold,
        "citacoes_esperadas": total_esperadas,
        "citacoes_detectadas": total_candidatos,
        "verdadeiros_positivos": verdadeiros_positivos,
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
        "extras_ignorados": extras_ignorados,
        "duplicatas_invalidas": duplicatas_invalidas,
        "submissao_valida": duplicatas_invalidas == 0,
        "spans_exatos": spans_exatos,
        "precisao": round(precisao, 4),
        "cobertura": round(cobertura, 4),
        "f1": round(f1, 4),
    }

# escolhe o threshold com base nos resultados no goldenset
def escolher_threshold(candidatos_por_documento, anotacoes_por_documento):
    thresholds = sorted({
        candidato["pontuacao"]
        for candidatos in candidatos_por_documento.values()
        for candidato in candidatos
    })
    if not thresholds:
        return 0, avaliar_conjunto(
            candidatos_por_documento, anotacoes_por_documento, 0
        )

    avaliacoes = [
        avaliar_conjunto(
            candidatos_por_documento, anotacoes_por_documento, threshold
        )
        for threshold in thresholds
    ]
    cobertura_total = [
        avaliacao for avaliacao in avaliacoes if avaliacao["cobertura"] == 1.0
    ]

    if cobertura_total:
        melhor = max(cobertura_total, key=lambda item: item["threshold"])
    else:
        melhor = max(
            avaliacoes,
            key=lambda item: (
                item["f1"], item["cobertura"], item["threshold"]
            ),
        )
    return melhor["threshold"], melhor


def ler_anotacoes(caminho=goldenset):
    with Path(caminho).open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def analisar_documentos(caminho_goldenset=goldenset, pasta_documentos=documentos,):
    anotacoes = ler_anotacoes(caminho_goldenset)
    anotacoes_por_documento = {}
    for anotacao in anotacoes:
        anotacoes_por_documento.setdefault(
            anotacao["documento_id"], []).append(anotacao)

    candidatos_por_documento = {}
    for documento_id in sorted(anotacoes_por_documento):
        caminho = Path(pasta_documentos) / f"{documento_id}.txt"
        texto_original = caminho.read_text(encoding="utf-8")
        tokens = criar_tokens(texto_original)
        registros = detectar_marcadores_e_indicios(texto_original, tokens)
        grupos = agrupar_registros(texto_original, registros)
        candidatos_por_documento[documento_id] = formar_candidatos(
            texto_original, tokens, grupos)

    threshold, treinamento = escolher_threshold( candidatos_por_documento, anotacoes_por_documento)

    documentos = []
    falsos_positivos_por_documento = []
    for documento_id, anotacoes_documento in sorted(
        anotacoes_por_documento.items()
    ):
        candidatos = [
            candidato
            for candidato in candidatos_por_documento[documento_id]
            if candidato["pontuacao"] >= threshold
        ]
        avaliacao = avaliar_documento(candidatos, anotacoes_documento)
        indices_correspondentes = avaliacao.pop("indices_correspondentes")
        indices_extras_ignorados = avaliacao.pop("indices_extras_ignorados")
        indices_ignorados = indices_correspondentes | indices_extras_ignorados
        falsos_positivos = [
            candidato for indice, candidato in enumerate(candidatos)
            if indice not in indices_ignorados]

        documentos.append({
            "documento_id": documento_id,
            "citacoes": candidatos,
            "avaliacao": avaliacao,
        })
        if falsos_positivos:
            falsos_positivos_por_documento.append({
                "documento_id": documento_id,
                "quantidade": len(falsos_positivos),
                "ocorrencias": falsos_positivos,
            })

    treinamento["falsos_positivos_por_documento"] = (
        falsos_positivos_por_documento)
    return {
        "treinamento_threshold": treinamento,
        "documentos": documentos,
    }


def main():
    parser = argparse.ArgumentParser(description="Detecta citações nos documentos e avalia contra o goldenset.")
    parser.add_argument("--goldenset", type=Path, default=goldenset)
    parser.add_argument("--txt", type=Path, default=documentos, help="pasta com os <documento_id>.txt")
    parser.add_argument("--saida", type=Path, help="grava o JSON neste arquivo em vez de imprimir")
    args = parser.parse_args()

    resposta = analisar_documentos(args.goldenset, args.txt)
    saida = json.dumps(resposta, ensure_ascii=False, indent=2)
    if args.saida is None:
        print(saida)
        return
    args.saida.write_text(saida, encoding="utf-8")
    t = resposta["treinamento_threshold"]
    #resumo no terminal; o detalhe fica no arquivo
    print({k: v for k, v in t.items() if k != "falsos_positivos_por_documento"})


if __name__ == "__main__":
    main()
