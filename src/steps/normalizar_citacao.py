"""Extrai atributos semânticos sem alterar o span ou o trecho original."""

from __future__ import annotations

import re
import unicodedata

import eda_utils as u
import indice_utils as ind
from configuracao import carregar_config
from modelos import CitacaoNormalizada, Familia
from steps.normalizar_numeros import normalizar
from steps.pipeline_utils import ANO_SOLTO, BLOCO_NUMERICO, corrigir_ocr, tem_identificador
from steps.rotear_citacao import E_SUMULA, E_TEMA

_TRIBUNAL = re.compile(
    r"\b(?:" + "|".join(map(re.escape, carregar_config("tribunais")["siglas"])) + r")\b", re.I
)
_UF = ind.UF_CITADA
_CONFIG_TRIBUNAIS = carregar_config("tribunais")
_TRIBUNAIS_CONHECIDOS = {sigla.upper() for sigla in _CONFIG_TRIBUNAIS["siglas"]}
_UFS_CONHECIDAS = {uf.upper() for uf in _CONFIG_TRIBUNAIS["ufs"]}
_ORGAO_EXPLICITO = re.compile(
    r"(?<!\w)(?:"
    r"S\.?\s*T\.?\s*-?\s*[A-Z]\.?|"
    r"T\.?\s*S\.?\s*-?\s*[A-Z]\.?|"
    r"T\.?\s*J\.?\s*-?\s*[A-Z]\.?\s*[A-Z]\.?|"
    r"T\.?\s*R\.?\s*[FT]\.?\s*-?\s*\d{1,3}|"
    r"T\.?\s*R\.?\s*E\.?\s*-?\s*[A-Z]\.?\s*[A-Z]\.?|"
    r"T\.?\s*[JR]\.?\s*M\.?\s*-?\s*[A-Z]\.?\s*[A-Z]\.?)"
    r"(?!\w)",
    re.I,
)
_UF_APOS_NUMERO = re.compile(
    r"(?<=\d)\s*(?P<separador>[/(–—-])\s*"
    r"(?P<sigla>[A-Za-z]{2,5})\)?(?=\b)"
)
_MARCADOR_NUMERO = re.compile(
    r"\b(?:processo|autos?)?\s*"
    r"(?:n(?:[uú]mero|[.º°o](?:\s*[º°o.])?)?\s*)",
    re.I,
)
_LIGACAO_QUALIFICADOR = re.compile(
    r"\s*(?:(?:,|e)\s*)?"
    r"(?:(?:do|da|dos|das|no|na|nos|nas|pelo|pela)\s+)?",
    re.I,
)
_UF_DE_CONTEXTO = re.compile(
    r"(?:[/(–—-]\s*)(?P<sigla>[A-Za-z]{2,5})\)?"
)
_NOME_LEI_A_DIREITA = re.compile(
    r"^\s+(?P<nome>d(?:a|e|o|as|os)\s+"
    r"[A-ZÀ-Þ][A-Za-zÀ-ÿ'-]*"
    r"(?:\s+(?:d(?:a|e|o|as|os)|à|às|e|[A-ZÀ-Þ][A-Za-zÀ-ÿ'-]*)){0,5})",
    re.I,
)
_ARTIGO = re.compile(
    r"\bart(?:igo)?\.?\s*(?P<numero>\d(?:[\d.]|\s+(?=\d))*)"
    r"(?:\s*(?-i:[º°o]))?(?:\s*[-–]\s*(?P<sufixo>[A-Za-z]+))?"
    r"(?![A-Za-z0-9])",
    re.I,
)
# Qualificador de ente federativo entre a espécie e o número: "Lei municipal nº
# 3.918/2016", "Lei Complementar estadual nº 812/2021", "Lei Mun. 2.741/19".
_ENTE_FEDERATIVO = r"(?:municipal|mun\.?|estadual|est\.?|distrital|dist\.?|federal|fed\.?|nacional)"
# O catálogo de diplomas só tem leis federais: uma lei estadual, municipal ou
# distrital de mesmo número e ano não é o diploma do catálogo.
_ENTES_NAO_FEDERAIS = {"municipal", "mun", "estadual", "est", "distrital", "dist"}
_IDENTIDADE_LEI = re.compile(
    r"\b(?:lei(?:\s+complementar)?|decreto\s*[-–]?\s*lei|lc)"
    rf"(?:\s+(?P<ente>{_ENTE_FEDERATIVO})(?![A-Za-zÀ-ÿ]))?\s*"
    r"(?:n(?:[uú]mero|[.º°o](?:\s*[º°o.])?)?\s*)?"
    r"(?P<numero>\d(?:[\d.\s]*\d)?)",
    re.I,
)
_ANO_APOS_BARRA = re.compile(r"^\s*/\s*(\d+)")
_ANO_APOS_DE = re.compile(r"^\s*,?\s*de\b([^.;:\n]{0,60})", re.I)
_LEI_NOMEADA = re.compile(r"\blei\s+d(?:a|e|o|as|os)\s+[A-Za-zÀ-ÿ]", re.I)
_CONTEXTO_LEGAL_FORTE = re.compile(
    r"\b(?:nos?\s+termos?|com\s+fundamento|conforme|previst[oa]s?|dispõe|aplica-se)\b",
    re.I,
)
_ANO = re.compile(r"\b(?:19|20)\d{2}\b")
_DATA_NUMERICA = re.compile(
    r"(?<![\d./–—-])(?:"
    r"\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2}|"
    r"(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}"
    r")(?![\d./–—-])"
)
_VINCULANTE = re.compile(r"vincul|\bSV\b(?=\s+\d)", re.I)
_MODIFICADORES_PROCESSUAIS = {
    "AG", "AGR", "AGRG", "AGINT", "ED", "EDCL", "EDS", "EDV",
    "EI", "EDIV", "ERESP", "EARESP", "QO",
}
_CONFIG_DIPLOMAS = carregar_config("diplomas")
_APELIDOS = _CONFIG_DIPLOMAS["apelidos_estatuto"]
_IDENTIDADES_NUMERADAS = {
    tuple(chave.split("/", 1)): diploma
    for chave, diploma in _CONFIG_DIPLOMAS["identidade_numerada_estatuto"].items()
}


def _numero_apos(trecho: str, palavra: re.Pattern) -> str | None:
    marcador = palavra.search(trecho)
    resto = corrigir_ocr(trecho[marcador.end():] if marcador else trecho)
    m = BLOCO_NUMERICO.search(resto)
    if not m:
        return None
    return re.sub(r"\D", "", m.group()) or None


def _sem_acento(texto: str) -> str:
    """Remove acentos sem alterar as posições dos caracteres-base."""
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def _canonizar_numero_acordao(fragmento: str) -> str | None:
    formas = u.variantes_numero(fragmento)
    if not formas:
        return None
    cnj = u.chave_cnj(formas[0])
    if cnj is not None:
        return cnj
    digitos = re.sub(r"\D", "", formas[0])
    return digitos or None


def _numero_acordao(trecho: str) -> str | None:
    """Seleciona o identificador associado à classe, não um ano de contexto."""
    sem_acento = _sem_acento(trecho)
    classes = [
        achado
        for padrao, _classe in ind.CLASSES_CITADAS
        for achado in padrao.finditer(sem_acento)
    ]
    # A classe-base costuma ser a ocorrência mais à direita em cadeias como
    # "AgInt no REsp". Procurar a partir dela também ignora anos introdutórios.
    for achado in sorted(classes, key=lambda item: item.start(), reverse=True):
        sufixo = trecho[achado.end():]
        marcador = _MARCADOR_NUMERO.search(_sem_acento(sufixo))
        if marcador:
            numero = _canonizar_numero_acordao(sufixo[marcador.end():])
            if numero is not None:
                return numero
        numero = _canonizar_numero_acordao(ANO_SOLTO.sub(" ", sufixo))
        if numero is not None:
            return numero

    marcador = _MARCADOR_NUMERO.search(sem_acento)
    if marcador:
        numero = _canonizar_numero_acordao(trecho[marcador.end():])
        if numero is not None:
            return numero

    # Sem uma classe explícita, ainda aceitamos um CNJ ou número isolado. Anos
    # introduzidos por "de/em" são removidos antes desse último recurso.
    return _canonizar_numero_acordao(
        re.sub(r"\b(?:de|em)\s+(?:19|20)\d{2}\b", " ", trecho, flags=re.I)
    )


def _remover_anos_contextuais(trecho: str) -> str:
    """Remove anos de contexto antes de normalizar blocos numéricos.

    A normalização agrupa dígitos separados por espaços. Sem esta etapa,
    ``REsp 2023 1.741.784`` viraria um único identificador. Anos que fazem
    parte de CNJ, datas ou que são o único número da citação são preservados.
    """
    limpo = _DATA_NUMERICA.sub(lambda m: " " * len(m.group()), trecho)
    limpo = ANO_SOLTO.sub(lambda m: " " * len(m.group()), limpo)
    for ano in reversed(list(_ANO.finditer(limpo))):
        anterior = limpo[ano.start() - 1] if ano.start() else ""
        posterior = limpo[ano.end()] if ano.end() < len(limpo) else ""
        if anterior in "./–—-" or posterior in "./–—-":
            continue
        janela = limpo[ano.end():ano.end() + 64]
        tem_identificador_depois = any(
            len(re.sub(r"\D", "", bloco.group())) >= 3
            for bloco in BLOCO_NUMERICO.finditer(janela)
        )
        if tem_identificador_depois:
            limpo = limpo[:ano.start()] + " " * len(ano.group()) + limpo[ano.end():]
    return limpo


def _e_modificador_processual(achado: re.Match) -> bool:
    """Distingue sufixos recursais de UFs escritas após hífen."""
    separador = achado.group("separador")
    sigla = achado.group("sigla").upper()
    return separador in "-–—" and sigla in _MODIFICADORES_PROCESSUAIS


def _qualificador_invalido(trecho: str) -> str | None:
    """Retorna um tribunal/UF explícito que não pertence ao catálogo."""
    tribunais = set()
    for achado in _ORGAO_EXPLICITO.finditer(trecho):
        sigla = re.sub(r"[^A-Za-z0-9]", "", achado.group()).upper()
        if sigla not in _TRIBUNAIS_CONHECIDOS:
            return achado.group()
        tribunais.add(sigla)
    if len(tribunais) > 1:
        return "/".join(sorted(tribunais))

    ufs = []
    for achado in _UF.finditer(trecho):
        sigla = achado.group(1).upper()
        if sigla not in _UFS_CONHECIDAS:
            return achado.group(1)
        ufs.append(sigla)
    for achado in _UF_APOS_NUMERO.finditer(trecho):
        sigla = achado.group("sigla").upper()
        if sigla in _TRIBUNAIS_CONHECIDOS:
            continue
        if _e_modificador_processual(achado):
            continue
        if sigla not in _UFS_CONHECIDAS:
            return achado.group("sigla")
        ufs.append(sigla)
    if len(set(ufs)) > 1:
        return "/".join(sorted(set(ufs)))

    return None


def _uf_explicita(trecho: str) -> str | None:
    """Extrai UF válida mesmo quando o hífen não tem espaços."""
    uf = _UF.search(trecho)
    if uf and uf.group(1).upper() in _UFS_CONHECIDAS:
        return uf.group(1).upper()
    for achado in _UF_APOS_NUMERO.finditer(trecho):
        sigla = achado.group("sigla").upper()
        if sigla in _UFS_CONHECIDAS:
            return sigla
    return None


def _tribunal_explicito(trecho: str) -> str | None:
    """Extrai também grafias pontuadas de tribunais conhecidos."""
    tribunal = _TRIBUNAL.search(trecho)
    if tribunal:
        return tribunal.group().upper()
    for achado in _ORGAO_EXPLICITO.finditer(trecho):
        sigla = re.sub(r"[^A-Za-z0-9]", "", achado.group()).upper()
        if sigla in _TRIBUNAIS_CONHECIDOS:
            return sigla
    return None


def _qualificadores_a_direita(contexto: str) -> str:
    """Recorta somente qualificadores ancorados à borda direita do span."""
    posicao = 0
    fim = 0
    for _ in range(3):
        ligacao = _LIGACAO_QUALIFICADOR.match(contexto, posicao)
        if not ligacao:
            break
        inicio = ligacao.end()
        orgao = _ORGAO_EXPLICITO.match(contexto, inicio)
        if orgao:
            posicao = fim = orgao.end()
            continue
        uf = _UF_DE_CONTEXTO.match(contexto, inicio)
        if uf:
            posicao = fim = uf.end()
            continue
        break
    return contexto[:fim] if fim else ""


def _qualificador_a_esquerda(contexto: str) -> str:
    """Recorta um órgão imediatamente anterior ao span, sem ampliar o span."""
    candidatos = [
        achado
        for achado in _ORGAO_EXPLICITO.finditer(contexto)
        if re.fullmatch(r"\s*[,:]?\s*", contexto[achado.end():])
    ]
    if candidatos:
        return candidatos[-1].group()
    return ""


def _trecho_com_contexto_de_qualificadores(candidato: dict) -> str:
    esquerda = _qualificador_a_esquerda(candidato.get("_contexto_esquerda", ""))
    direita = _qualificadores_a_direita(candidato.get("_contexto_direita", ""))
    return " ".join(parte for parte in (esquerda, candidato["trecho"], direita) if parte)


def _trecho_com_contexto_legal(candidato: dict) -> str:
    """Inclui nome de lei imediatamente cortado à direita, sem mudar o span."""
    trecho = candidato["trecho"]
    if not re.search(r"\blei\s*$", trecho, re.I):
        return trecho
    nome = _NOME_LEI_A_DIREITA.match(candidato.get("_contexto_direita", ""))
    return trecho + (" " + nome.group("nome") if nome else "")


def _normalizar_lexico(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


def _diploma(trecho: str) -> str | None:
    texto = _normalizar_lexico(trecho)
    correspondencias = []
    for nome, apelidos in _APELIDOS.items():
        for apelido in apelidos:
            alias = _normalizar_lexico(apelido)
            if alias and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", texto):
                correspondencias.append((len(alias), nome))
    return max(correspondencias, default=(0, None))[1]


def _artigo(trecho: str) -> str | None:
    # Prefere a leitura com mais dígitos. Assim, um prefixo parcial como ``3``
    # em ``3 l 2`` não impede a leitura OCR de ``312``, e empates preservam a
    # superfície original e seus sufixos jurídicos (``290-B``).
    bruto = _ARTIGO.search(trecho)
    corrigido = _ARTIGO.search(corrigir_ocr(trecho))
    candidatos = [artigo for artigo in (bruto, corrigido) if artigo]
    if not candidatos:
        return None
    artigo = max(
        candidatos,
        key=lambda item: len(re.sub(r"\D", "", item.group("numero"))),
    )
    numero = re.sub(r"\D", "", artigo.group("numero"))
    sufixo = artigo.group("sufixo")
    return f"{numero}-{sufixo.upper()}" if sufixo else numero


def _normalizar_ano_lei(ano: str | None) -> str | None:
    if ano is None:
        return None
    if len(ano) == 2:
        return ("19" if int(ano) >= 30 else "20") + ano
    return ano


def _ano_lei_apos(trecho: str) -> str | None:
    por_barra = _ANO_APOS_BARRA.match(trecho)
    if por_barra:
        return _normalizar_ano_lei(por_barra.group(1))
    por_data = _ANO_APOS_DE.match(trecho)
    if not por_data:
        return None
    anos = re.findall(r"\b(?:19|20)\d{2}\b", por_data.group(1))
    return anos[0] if anos else None


def _identidades_lei(trecho: str) -> list[tuple[str, str | None, bool]]:
    """(número, ano, ente não federal) de cada lei numerada do trecho."""
    corrigido = corrigir_ocr(trecho)
    return [
        (
            re.sub(r"\D", "", achado.group("numero")),
            _ano_lei_apos(corrigido[achado.end():]),
            (achado.group("ente") or "").lower().rstrip(".") in _ENTES_NAO_FEDERAIS,
        )
        for achado in _IDENTIDADE_LEI.finditer(corrigido)
    ]


def _diploma_por_identidade(numero: str, ano: str | None) -> str | None:
    candidatos = {
        diploma
        for (numero_canonico, ano_canonico), diploma in _IDENTIDADES_NUMERADAS.items()
        if numero == numero_canonico and (ano is None or ano == ano_canonico)
    }
    return next(iter(candidatos)) if len(candidatos) == 1 else None


def _dados_lei(trecho: str) -> tuple[str | None, str | None, str | None, bool | None, bool]:
    diploma_nomeado = _diploma(trecho)
    identidades = _identidades_lei(trecho)
    if not identidades:
        return diploma_nomeado, None, None, None, False

    diplomas_numerados = [
        None if nao_federal else _diploma_por_identidade(numero, ano)
        for numero, ano, nao_federal in identidades
    ]
    identidade_valida = (
        all(diplomas_numerados)
        and len(set(diplomas_numerados)) == 1
        and (not diploma_nomeado or diploma_nomeado == diplomas_numerados[0])
    )
    diploma = diplomas_numerados[0] if identidade_valida else diploma_nomeado
    numero, ano, _ = identidades[0]
    return diploma, numero, ano, identidade_valida, True


def normalizar_citacao(candidato: dict, familia: Familia, *, ja_normalizado: bool = False) -> CitacaoNormalizada:
    trecho = candidato["trecho"]
    texto_normalizado = trecho if ja_normalizado else normalizar(trecho, familia)
    trecho_qualificadores = _trecho_com_contexto_de_qualificadores(candidato)
    trecho_legal = _trecho_com_contexto_legal(candidato) if familia == "lei" else trecho
    # Tribunal é texto, não número. Corrigir OCR aqui podia transformar o
    # ``ST`` de ``STJ`` em ``57`` quando a sigla vinha logo depois do processo.
    tribunal = _tribunal_explicito(trecho_qualificadores)
    qualificador_invalido = (
        _qualificador_invalido(trecho_qualificadores)
        if familia in {"acordao", "sumula", "tema"}
        else None
    )
    uf = _uf_explicita(trecho_qualificadores)
    ano = _ANO.search(trecho)
    artigo = _artigo(trecho_legal) if familia == "lei" else None
    if familia == "lei":
        diploma, numero_lei, ano_lei, identidade_legal_valida, lei_numerada = _dados_lei(trecho_legal)
        lei_nomeada = bool(_LEI_NOMEADA.search(trecho_legal))
        contexto_legal_forte = bool(artigo or _CONTEXTO_LEGAL_FORTE.search(trecho_legal))
    else:
        diploma = numero_lei = ano_lei = None
        identidade_legal_valida = None
        lei_numerada = lei_nomeada = contexto_legal_forte = False
    numero = None
    identificador_adulterado = False
    if familia == "sumula":
        numero = _numero_apos(trecho, E_SUMULA)
    elif familia == "tema":
        numero = _numero_apos(trecho, E_TEMA)
    elif familia == "acordao":
        trecho_para_numero = _remover_anos_contextuais(trecho)
        texto_para_numero = (
            trecho_para_numero
            if ja_normalizado
            else normalizar(trecho_para_numero, familia)
        )
        numero = _numero_acordao(texto_para_numero)
        # Sem número selecionado e com estrutura de CNJ no trecho: o CNJ foi
        # rejeitado por dígito a mais, sufixo ou prefixo. É adulteração, não
        # ausência de identificador.
        identificador_adulterado = numero is None and u.cnj_adulterado(texto_para_numero)
    return CitacaoNormalizada(
        familia=familia,
        inicio=candidato["inicio"],
        fim=candidato["fim"],
        trecho=trecho,
        normalizado=texto_normalizado,
        tribunal=tribunal,
        qualificador_invalido=qualificador_invalido,
        classe=ind._classe_citada(trecho) if familia == "acordao" else None,
        numero=numero,
        uf=uf,
        ano=ano.group() if ano else None,
        artigo=artigo,
        diploma=diploma,
        numero_lei=numero_lei,
        ano_lei=ano_lei,
        identidade_legal_valida=identidade_legal_valida,
        lei_nomeada=lei_nomeada,
        contexto_legal_forte=contexto_legal_forte,
        vinculante=bool(_VINCULANTE.search(trecho)) if familia == "sumula" else False,
        lei_numerada=lei_numerada,
        identificador_suficiente=tem_identificador(trecho),
        identificador_adulterado=identificador_adulterado,
        fonte=candidato.get("fonte", "regra"),
    )
