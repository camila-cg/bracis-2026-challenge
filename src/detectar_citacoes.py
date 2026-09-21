
""""
marcador (de vocabulário) -> termo do vocabulário que será procurado
marcador principal   -> ocorrência desse termo no texto
indício (complementar)  ->  informação próxima ao marcador principal, como número, tribunal, UF, relatoria, ano ou nome próprio (espera-se wue de pessoa)
registro -> guarda indícios e marcadores
citação -> grupo formado por um marcador principal e seus indícios complementares.
"""

import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


caminho = Path(__file__).resolve().parent / "desafio-jusbrasil-bracis-2026"
goldenset = caminho / "goldenset_offsets.csv"
documentos = caminho / "txt"


# Vocabulário principal, baseado no goldenset.
#BoW nas citações do gabarito + complementação com IA
MARCADORES_VOCABULARIO = {
    "jurisprudencia": [
        "reclamacao", "recl", "rcl",
        "recurso", "recurso especial", "re", "resp", "aresp", "agresp",
        "recurso em mandado", "recurso em mandado de seguranca",
        "recurso em habeas corpus", "rhc", "respe", "rec esp", "r esp",
        "agravo interno", "agint", "agravo regimental", "agrg",
        "agravo de instrumento", "agravo em recurso especial", "agravo",
        "ag int", "ag instr", "ag reg",
        "habeas corpus", "h c", "hc", "rse", "apl", "rms", "rr", "arr", "ar",
        "embargos de declaracao", "ed", "edcl", "eds",
        "sumula", "sum", "sumula vinculante", "tema", "repercussao geral",
        "processo", "acordao", "julgado", "precedente", "agr"],
    "lei": [ "art", "artigo", "lei", "lei complementar", "codigo",
        "constituicao", "constituicao federal", "constituicao da republica",
        "clt", "cpc", "codigo civil", "processo civil", "processo penal",
        "defesa do consumidor", "codigo penal", "penal militar",
        "codigo eleitoral", "consolidacao das leis do trabalho", "§",],
}

TRIBUNAIS = {"stf", "stj", "tst", "tse", "stm"}

UFS = {"ac", "al", "ap", "am", "ba", "ce", "df", "es", "go", "ma",
    "mt", "ms", "mg", "pa", "pb", "pr", "pe", "pi", "rj", "rn",
    "rs", "ro", "rr", "sc", "sp", "se", "to",}

TERMOS_RELATORIA = {"rel", "relator", "relatora", "relatoria", "min", "ministro", "ministra",}

COMPOSICAO_NOME = {"da", "das", "de", "do", "dos", "e"}

MARCADORES_DESCRITIVOS = {"acordao", "julgado", "precedente", "reclamacao"}
MARCADORES_NUMERADOS = {"sumula", "sum", "sumula vinculante", "tema"}
MARCADORES_LEGAIS_NUMERADOS = { "art", "artigo", "§", "lei", "lei complementar",}

SIGLAS = { "re", "resp", "aresp", "respe", "rhc", "hc", "rse", "apl", "rms",
    "rr", "arr", "ar", "rp", "agarr", "rcl", "recl", "agr", "agrg",
    "agint", "ed", "edcl", "eds",}

#para evitar que confuda com marcadores principais
TERMOS_PRIORITARIOS_INDICIOS = (TERMOS_RELATORIA | TRIBUNAIS)

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


def procurar_marcadores_principais(tokens):
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

         #agora somente procura no vocabulário se não for uma sigla composta
        if marcador_encontrado is None:
            for tamanho in (4, 3, 2, 1):
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
                    #expressão similar
                    marcador, similaridade = encontrar_marcador_parecido( expressao, marcadores_mesmo_tamanho)
                    if marcador is not None:
                        marcador_encontrado = (categoria, marcador, similaridade, fim - 1)
                        break

                if marcador_encontrado is not None:
                    break
        if marcador_encontrado is None:
            continue
        categoria, marcador, similaridade, fim_indice = marcador_encontrado
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


def procurar_numeros(tokens):
    indicios = []

    for indice, token in enumerate(tokens):
        texto = token["normalizado"]
        if not re.search(r"\d", texto):
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", texto):
            indicios.append(criar_registro("ano", indice, indice, tokens, 1, valor=texto))
            continue

        quantidade_digitos = len(re.findall(r"\d", texto))
        possui_formato_processual = (quantidade_digitos >= 4 or bool(re.search(r"\d[./-]\d", texto)) or bool(re.search(r"[a-z]+-.*\d", texto)))

        tipo = "numero_processual" if possui_formato_processual else "numero_legal"
        pontos=1
        indicios.append(criar_registro(tipo, indice, indice, tokens, pontos, valor=texto))

    return indicios


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

######### Essa parte debaixo foi totalmente feita por IA (e não toda foi revisada a fundo) para conseguir executar o código e testar o funcionamento das regras acima, ajustar algumas coisas... Ainda não fiz a parte oficial 
####### de juntar os registros identificados em citações, calcular a pontuação final, comparar com os documentos e dar um retorno
####### parece que está funcionando, mas tem que ser revisada e pelo que olhei não era exatamente como eu estava pensando, mas deve dar para aproveitar algumas coisas mais mecânicas e que não vale a pena perder muito tempo fazendo

#essa já foi revisada, está ok
def detectar_marcadores_e_indicios(texto_original, tokens):
    registros = []
    registros.extend(procurar_marcadores_principais(tokens))
    registros.extend(procurar_numeros(tokens))
    registros.extend(procurar_tribunais_e_ufs(tokens))
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
        marcadores_do_grupo = []

        for item in grupo_atual:
            if item["tipo_registro"] == "marcador_principal":
                marcadores_do_grupo.append(item)

        novo_marcador_distante = False
        if registro["tipo_registro"] == "marcador_principal" and marcadores_do_grupo:
            ultimo_marcador = marcadores_do_grupo[-1]
            distancia_marcadores = (registro["inicio_token"] - ultimo_marcador["fim_token"] - 1)
            #heurística para quando tem mais de uma citação na mesma frase/parágrafo. Parece boa
            novo_marcador_distante = (distancia_marcadores > 2 and registro.get("categoria") == "jurisprudencia")

        #outra heurística que talvez precise ser ajustada para generalização
        if (distancia <= 8 and tamanho_grupo <= 24 and not novo_marcador_distante and not limite_entre_citacoes(texto_original, anterior, registro)):
            grupo_atual.append(registro)
        else:
            grupos.append(grupo_atual)
            grupo_atual = [registro]
    #salva o último grupo
    if grupo_atual:
        grupos.append(grupo_atual)
    return grupos


def escolher_categoria(grupo):
    marcadores_principais = [
        registro for registro in grupo
        if registro["tipo_registro"] == "marcador_principal"
    ]
    if not marcadores_principais:
        return None

    marcadores_lei = {
        registro["marcador"] for registro in marcadores_principais
        if registro["categoria"] == "lei"
    }
    if marcadores_lei & MARCADORES_LEGAIS_NUMERADOS:
        return "lei"

    pontos_por_categoria = {"jurisprudencia": 0, "lei": 0}
    for registro in marcadores_principais:
        pontos_por_categoria[registro["categoria"]] += registro["pontos"]
    return max(pontos_por_categoria, key=pontos_por_categoria.get)


def grupo_atende_regras(grupo, categoria):
    marcadores_principais = [
        registro for registro in grupo
        if registro["tipo_registro"] == "marcador_principal"
        and registro["categoria"] == categoria
    ]
    marcadores = {
        registro["marcador"] for registro in marcadores_principais
    }
    tipos = {registro["tipo_registro"] for registro in grupo}

    if categoria == "lei":
        tem_marcador_legal_numerado = bool(
            marcadores & MARCADORES_LEGAIS_NUMERADOS
        )
        tem_numero = bool(tipos & {"numero_legal", "numero_processual"})

        return tem_marcador_legal_numerado and tem_numero

    tem_numero_processual = "numero_processual" in tipos
    tem_algum_numero = bool(tipos & {"numero_processual", "numero_legal"})
    fragmentos_numericos = sum(
        registro["tipo_registro"] == "numero_legal"
        for registro in grupo
    )
    tem_descricao_completa = (
        "ano" in tipos
        and bool(tipos & {"relatoria", "nome_proprio"})
    )

    if marcadores & MARCADORES_NUMERADOS:
        return tem_algum_numero
    return (
        tem_numero_processual
        or fragmentos_numericos >= 2
        or tem_descricao_completa
    )


def pontuar_grupo(grupo, categoria):
    pontos = 0
    registros_contados = set()

    for registro in grupo:
        if registro["tipo_registro"] == "marcador_principal":
            if registro["categoria"] != categoria:
                continue
        elif categoria == "lei" and registro["tipo_registro"] in {
            "tribunal", "uf", "relatoria", "nome_proprio"
        }:
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


def formar_candidatos(texto_original, tokens, grupos):
    candidatos = []

    for grupo in grupos:
        categoria = escolher_categoria(grupo)
        if categoria is None or not grupo_atende_regras(grupo, categoria):
            continue

        registros_categoria = [
            registro for registro in grupo
            if registro["tipo_registro"] != "marcador_principal"
            or registro["categoria"] == categoria
        ]
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
                {
                    "tipo": registro["tipo_registro"],
                    "valor": registro.get("marcador", registro.get("valor")),
                    "pontos": registro["pontos"],
                }
                for registro in registros_categoria
            ],
        })

    return candidatos


def calcular_sobreposicao(candidato, anotacao):
    inicio = max(candidato["inicio"], int(anotacao["inicio"]))
    fim = min(candidato["fim"], int(anotacao["fim"]))
    intersecao = max(0, fim - inicio)
    if intersecao == 0:
        return 0

    uniao = (
        max(candidato["fim"], int(anotacao["fim"]))
        - min(candidato["inicio"], int(anotacao["inicio"]))
    )
    return intersecao / uniao


def avaliar_documento(candidatos, anotacoes):
    candidatos_usados = set()
    correspondencias = []

    for anotacao in anotacoes:
        melhor_indice = None
        melhor_sobreposicao = 0

        for indice, candidato in enumerate(candidatos):
            if indice in candidatos_usados:
                continue
            if candidato["tipo"] != anotacao["tipo"]:
                continue

            sobreposicao = calcular_sobreposicao(candidato, anotacao)
            if sobreposicao > melhor_sobreposicao:
                melhor_indice = indice
                melhor_sobreposicao = sobreposicao

        if melhor_indice is not None and melhor_sobreposicao > 0:
            candidatos_usados.add(melhor_indice)
            candidato = candidatos[melhor_indice]
            correspondencias.append({
                "citacao_id": anotacao["citacao_id"],
                "indice_candidato": melhor_indice,
                "sobreposicao": round(melhor_sobreposicao, 3),
                "span_exato": (
                    candidato["inicio"] == int(anotacao["inicio"])
                    and candidato["fim"] == int(anotacao["fim"])
                ),
            })

    verdadeiros_positivos = len(correspondencias)
    falsos_positivos = len(candidatos) - verdadeiros_positivos
    falsos_negativos = len(anotacoes) - verdadeiros_positivos
    precisao = (
        verdadeiros_positivos / len(candidatos) if candidatos else 0
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
        "spans_exatos": sum(item["span_exato"] for item in correspondencias),
        "precisao": round(precisao, 4),
        "cobertura": round(cobertura, 4),
        "f1": round(f1, 4),
        "indices_correspondentes": {
            item["indice_candidato"] for item in correspondencias
        },
    }


def avaliar_conjunto(candidatos_por_documento, anotacoes_por_documento, threshold):
    total_candidatos = 0
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0
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
        spans_exatos += avaliacao["spans_exatos"]

    total_esperadas = verdadeiros_positivos + falsos_negativos
    precisao = verdadeiros_positivos / total_candidatos if total_candidatos else 0
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
        "spans_exatos": spans_exatos,
        "precisao": round(precisao, 4),
        "cobertura": round(cobertura, 4),
        "f1": round(f1, 4),
    }


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


def analisar_documentos(
    caminho_goldenset=goldenset,
    pasta_documentos=documentos,
):
    anotacoes = ler_anotacoes(caminho_goldenset)
    anotacoes_por_documento = {}
    for anotacao in anotacoes:
        anotacoes_por_documento.setdefault(
            anotacao["documento_id"], []
        ).append(anotacao)

    candidatos_por_documento = {}
    for documento_id in sorted(anotacoes_por_documento):
        caminho = Path(pasta_documentos) / f"{documento_id}.txt"
        texto_original = caminho.read_text(encoding="utf-8")
        tokens = criar_tokens(texto_original)
        registros = detectar_marcadores_e_indicios(texto_original, tokens)
        grupos = agrupar_registros(texto_original, registros)
        candidatos_por_documento[documento_id] = formar_candidatos(
            texto_original, tokens, grupos
        )

    threshold, treinamento = escolher_threshold(
        candidatos_por_documento, anotacoes_por_documento
    )

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
        falsos_positivos = [
            candidato for indice, candidato in enumerate(candidatos)
            if indice not in indices_correspondentes
        ]

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
        falsos_positivos_por_documento
    )
    return {
        "treinamento_threshold": treinamento,
        "documentos": documentos,
    }


def main():
    resposta = analisar_documentos()
    print(json.dumps(resposta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
