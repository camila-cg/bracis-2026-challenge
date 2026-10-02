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
# Números de processos antigos podem ter somente um ou dois dígitos. Esse
# fallback é deliberadamente separado de ``NUMERO_LIVRE``: só o chamador que
# já confirmou classe processual + marcador explícito pode habilitá-lo.
_NUMERO_PROCESSUAL_CURTO = re.compile(
    r"^\s*(?P<numero>\d{1,2})(?!\d)(?!\s*[.\-–—]\s*\d)(?!\s*/\s*\d)"
)
# Classes cujo número curto vem logo depois da sigla, sem ``nº`` (``IAC 15/STJ``).
_CLASSES_NUMERO_CURTO_SEM_INDICADOR = set(
    carregar_config("classes_processuais").get("classes_numero_curto_sem_indicador", [])
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
# Marcador de artigo no singular ou no plural. Um ``s`` seguido de ponto ou
# espaço é sempre o plural: sem isso, ``arts. 5º`` passava pela correção de OCR
# como ``art5. 5º`` e era lido como art. 55.
_MARCADOR_ARTIGO = r"\bart(?:igo)?(?:s(?=[.\s])|(?!s[.\s]))\.?\s*"
_ARTIGO = re.compile(
    _MARCADOR_ARTIGO + r"(?P<numero>\d(?:[\d.]|\s+(?=\d))*)"
    r"(?:\s*(?-i:[º°o]))?(?:\s*[-–]\s*(?P<sufixo>[A-Za-z]+))?"
    r"(?![A-Za-z0-9])",
    re.I,
)
# Artigos coordenados logo após o marcador no plural: ``arts. 5º e 7º``,
# ``artigos 186, 187 e 927``. A lista termina no primeiro item que não é um
# número de artigo, então incisos e parágrafos (``arts. 5º, LIV e LV``,
# ``arts. 5º, § 1º``) não são lidos como artigos.
_ITEM_ARTIGO = (
    r"(?P<numero>\d(?:[\d.]*\d)?)(?:\s*(?-i:[º°o]))?"
    r"(?:\s*[-–]\s*(?P<sufixo>(?-i:[A-Z]))(?![A-Za-z]))?(?![\d/])"
)
_ITEM_ARTIGO_SEM_GRUPOS = re.sub(r"\(\?P<\w+>", "(?:", _ITEM_ARTIGO)
_LISTA_ARTIGOS = re.compile(
    r"\bart(?:igo)?s(?=[.\s])\.?\s*"
    rf"(?P<lista>{_ITEM_ARTIGO_SEM_GRUPOS}"
    rf"(?:\s*(?:,\s*(?:e\s+)?|e\s+){_ITEM_ARTIGO_SEM_GRUPOS})+)",
    re.I,
)
# Qualificador de ente federativo entre a espécie e o número: "Lei municipal nº
# 3.918/2016", "Lei Complementar estadual nº 812/2021", "Lei Mun. 2.741/19".
_ENTE_FEDERATIVO = r"(?:municipal|mun\.?|estadual|est\.?|distrital|dist\.?|federal|fed\.?|nacional)"
# O catálogo de diplomas só tem leis federais: uma lei estadual, municipal ou
# distrital de mesmo número e ano não é o diploma do catálogo.
_ENTES_NAO_FEDERAIS = {"municipal", "mun", "estadual", "est", "distrital", "dist"}
# Recupera "art." cortado à esquerda do span pelo extrator (ex. quando o número
# do artigo foi inteiramente substituído por letra de OCR e por isso não sobrou
# nenhum dígito para ancorar o agrupamento). Não altera o span nem o trecho
# reportado — mesmo princípio de _qualificador_a_esquerda. A janela pequena e a
# exigência de não cruzar ponto final limitam o risco de pegar um "art." de
# outra frase.
_ARTIGO_PREFIXO_ESQUERDA = re.compile(r"\bart(?:igo)?\.?\s*[^.\n]{0,30}$", re.I)
# Letra de OCR logo após "art."/"artigo" quando o contexto imediatamente
# seguinte (dígito, ponto+dígito, ou o sinal ordinal º/°) confirma que ali
# deveria haver um número. A âncora em "art." é o que torna seguro corrigir
# aqui mesmo sem dígito adjacente sobrevivente (diferente da correção de OCR
# genérica em corrigir_ocr, que exige adjacência a um dígito para não
# corromper palavras comuns como "artigo importante").
_LETRAS_OCR_ARTIGO = "".join(u.OCR_PARA_DIGITO)
_ARTIGO_OCR_INICIAL = re.compile(
    r"(?P<marcador>" + _MARCADOR_ARTIGO + r")(?P<letras>[" + re.escape(_LETRAS_OCR_ARTIGO) + r"]{1,2})"
    r"(?=[.\s]*\d|\s*[º°])",
    re.I,
)


def _prefixo_artigo_a_esquerda(contexto_esquerda: str) -> str:
    m = _ARTIGO_PREFIXO_ESQUERDA.search(contexto_esquerda)
    return m.group() if m else ""


def _corrigir_ocr_artigo(trecho: str) -> str:
    def _trocar(m: re.Match) -> str:
        corpo = "".join(u.OCR_PARA_DIGITO.get(c, c) for c in m.group("letras"))
        return m.group("marcador") + corpo
    return _ARTIGO_OCR_INICIAL.sub(_trocar, trecho)


_IDENTIDADE_LEI = re.compile(
    # `decreto-lei` vem antes de `decreto` (a alternância escolhe a primeira que casa).
    # `res.` exige o ponto, para não casar a palavra comum "res"; a sigla do órgão
    # emissor (`Res. CNJ nº 455`) é opcional e só vale em maiúsculas.
    r"\b(?:lei(?:\s+complementar)?|decreto\s*[-–]?\s*lei|decreto|decr\.?|resolu[çc][ãa]o|res\.|lc)"
    r"(?:\s+(?-i:[A-ZÀ-Ý]{2,8})(?![A-Za-zÀ-ÿ]))?"
    rf"(?:\s+(?P<ente>{_ENTE_FEDERATIVO})(?![A-Za-zÀ-ÿ]))?\s*"
    r"(?:n(?:[uú]mero|[.º°o](?:\s*[º°o.])?)?\s*)?"
    r"(?P<numero>\d(?:[\d.\s]*\d)?)",
    re.I,
)
# OCR que troca o "i" de "Lei" por "1" ou "l"; só vale com número logo depois.
_OCR_DE_LEI = re.compile(r"(?<![\w])Le[1l](?=\s*(?:n(?:[uú]mero|[.º°o])?\s*)?\d)", re.I)
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
_ENUNCIADO = re.compile(r"\benunciado\b", re.I)
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
# Leis conhecidas pelo nome e ausentes da base (``LGPD``, ``Lei Maria da
# Penha``): o apelido vale como o número/ano da lei, para que ``art. 7º da
# LGPD`` seja classificado igual a ``art. 7º da Lei nº 13.709/2018``.
_IDENTIDADE_POR_APELIDO = {
    apelido: tuple(chave.split("/", 1))
    for chave, apelidos in _CONFIG_DIPLOMAS["identidade_por_apelido_fora_da_base"].items()
    for apelido in apelidos
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


def _canonizar_numero_acordao(
    fragmento: str, *, aceitar_curto: bool = False
) -> str | None:
    formas = u.variantes_numero(fragmento)
    if not formas:
        if aceitar_curto:
            curto = _NUMERO_PROCESSUAL_CURTO.match(corrigir_ocr(fragmento))
            if curto:
                return curto.group("numero")
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
        (achado, classe)
        for padrao, classe in ind.CLASSES_CITADAS
        for achado in padrao.finditer(sem_acento)
    ]
    # A classe-base costuma ser a ocorrência mais à direita em cadeias como
    # "AgInt no REsp". Procurar a partir dela também ignora anos introdutórios.
    for achado, classe in sorted(classes, key=lambda item: item[0].start(), reverse=True):
        sufixo = trecho[achado.end():]
        marcador = _MARCADOR_NUMERO.search(_sem_acento(sufixo))
        if marcador:
            numero = _canonizar_numero_acordao(
                sufixo[marcador.end():], aceitar_curto=True
            )
            if numero is not None:
                return numero
        numero = _canonizar_numero_acordao(
            ANO_SOLTO.sub(" ", sufixo),
            aceitar_curto=classe in _CLASSES_NUMERO_CURTO_SEM_INDICADOR,
        )
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

    A busca roda sobre o texto com o OCR já corrigido (o comprimento não muda).
    Sem isso, o trecho ``ano.segmento.tribunal`` de um CNJ seguido de uma letra
    de OCR (``2025.7.0O.0000``) parece a data ``2025.7.0`` e é apagado.
    """
    limpo = _DATA_NUMERICA.sub(lambda m: " " * len(m.group()), corrigir_ocr(trecho))
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


def _apelido_mais_longo(trecho: str, apelidos_por_valor):
    texto = _normalizar_lexico(trecho)
    correspondencias = []
    for valor, apelidos in apelidos_por_valor:
        for apelido in apelidos:
            alias = _normalizar_lexico(apelido)
            if alias and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", texto):
                correspondencias.append((len(alias), valor))
    return max(correspondencias, default=(0, None))[1]


def _diploma(trecho: str) -> str | None:
    return _apelido_mais_longo(trecho, _APELIDOS.items())


def _identidade_por_apelido(trecho: str) -> tuple[str, str] | None:
    return _apelido_mais_longo(
        trecho,
        ((identidade, [apelido]) for apelido, identidade in _IDENTIDADE_POR_APELIDO.items()),
    )


def _artigo(trecho: str, contexto_esquerda: str = "") -> str | None:
    # Prefere a leitura com mais dígitos. Assim, um prefixo parcial como ``3``
    # em ``3 l 2`` não impede a leitura OCR de ``312``, e empates preservam a
    # superfície original e seus sufixos jurídicos (``290-B``).
    ampliado = _prefixo_artigo_a_esquerda(contexto_esquerda) + trecho
    bruto = _ARTIGO.search(ampliado)
    corrigido = _ARTIGO.search(corrigir_ocr(ampliado))
    ancorado = _ARTIGO.search(corrigir_ocr(_corrigir_ocr_artigo(ampliado)))
    candidatos = [artigo for artigo in (bruto, corrigido, ancorado) if artigo]
    if not candidatos:
        return None
    artigo = max(
        candidatos,
        key=lambda item: len(re.sub(r"\D", "", item.group("numero"))),
    )
    numero = re.sub(r"\D", "", artigo.group("numero"))
    sufixo = artigo.group("sufixo")
    return f"{numero}-{sufixo.upper()}" if sufixo else numero


def _artigos_coordenados(trecho: str, artigo: str | None) -> tuple[str, ...]:
    """Artigos citados junto com o primeiro em ``arts. 5º e 7º``.

    O span continua único e o id é o do primeiro artigo; os demais só servem
    para conferir se todos existem na base.
    """
    if artigo is None:
        return ()
    lista = _LISTA_ARTIGOS.search(corrigir_ocr(trecho))
    if lista is None:
        return ()
    artigos = []
    for item in re.finditer(_ITEM_ARTIGO, lista.group("lista")):
        numero = re.sub(r"\D", "", item.group("numero"))
        sufixo = item.group("sufixo")
        lido = f"{numero}-{sufixo.upper()}" if sufixo else numero
        if lido != artigo and lido not in artigos:
            artigos.append(lido)
    return tuple(artigos)


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
    corrigido = _OCR_DE_LEI.sub(lambda m: m.group()[:2] + "i", corrigir_ocr(trecho))
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


def _numero_de_lei_conhecido(numero: str) -> bool:
    return any(numero == numero_canonico for numero_canonico, _ in _IDENTIDADES_NUMERADAS)


def _dados_lei(trecho: str) -> tuple[str | None, str | None, str | None, bool | None, bool]:
    diploma_nomeado = _diploma(trecho)
    identidades = _identidades_lei(trecho)
    if not identidades:
        apelido = None if diploma_nomeado else _identidade_por_apelido(trecho)
        if apelido:
            numero, ano = apelido
            # A identidade é suficiente para consultar uma base nova. Ela não
            # é inválida só porque estava ausente do catálogo antigo.
            return None, numero, ano, None, True
        return diploma_nomeado, None, None, None, False

    diplomas_numerados = []
    identidade_invalida = False
    for numero, ano, nao_federal in identidades:
        diploma = None if nao_federal else _diploma_por_identidade(numero, ano)
        diplomas_numerados.append(diploma)
        identidade_invalida |= nao_federal or (
            _numero_de_lei_conhecido(numero) and diploma is None
        )

    diplomas_conhecidos = {diploma for diploma in diplomas_numerados if diploma}
    if len(identidades) > 1 and (
        None in diplomas_numerados or len(diplomas_conhecidos) != 1
    ):
        identidade_invalida = True
    if diploma_nomeado and (
        len(diplomas_conhecidos) != 1 or diploma_nomeado not in diplomas_conhecidos
    ):
        identidade_invalida = True

    if identidade_invalida:
        identidade_valida = False
    elif len(diplomas_conhecidos) == 1:
        identidade_valida = True
    else:
        # Lei federal ainda desconhecida pela configuração: a base canônica
        # decide. ``None`` significa não verificada, não inválida.
        identidade_valida = None
    diploma = next(iter(diplomas_conhecidos), diploma_nomeado)
    numero, ano, _ = identidades[0]
    return diploma, numero, ano, identidade_valida, True


def normalizar_citacao(candidato: dict, familia: Familia, *, ja_normalizado: bool = False) -> CitacaoNormalizada:
    trecho_original = candidato["trecho"]
    # A análise usa NFC para que regexes tratem acentos compostos e
    # decompostos igualmente; o dataclass preserva a superfície e os offsets.
    trecho = unicodedata.normalize("NFC", trecho_original)
    texto_normalizado = trecho if ja_normalizado else normalizar(trecho, familia)
    trecho_qualificadores = unicodedata.normalize(
        "NFC", _trecho_com_contexto_de_qualificadores(candidato)
    )
    trecho_legal = (
        unicodedata.normalize("NFC", _trecho_com_contexto_legal(candidato))
        if familia == "lei" else trecho
    )
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
    artigo = (
        _artigo(trecho_legal, candidato.get("_contexto_esquerda", ""))
        if familia == "lei" else None
    )
    artigos_coordenados = _artigos_coordenados(trecho_legal, artigo) if familia == "lei" else ()
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
        trecho=trecho_original,
        normalizado=texto_normalizado,
        tribunal=tribunal,
        qualificador_invalido=qualificador_invalido,
        classe=ind._classe_citada(trecho) if familia == "acordao" else None,
        numero=numero,
        uf=uf,
        ano=ano.group() if ano else None,
        artigo=artigo,
        artigos_coordenados=artigos_coordenados,
        diploma=diploma,
        numero_lei=numero_lei,
        ano_lei=ano_lei,
        identidade_legal_valida=identidade_legal_valida,
        lei_nomeada=lei_nomeada,
        contexto_legal_forte=contexto_legal_forte,
        vinculante=bool(_VINCULANTE.search(trecho)) if familia == "sumula" else False,
        enunciado=bool(_ENUNCIADO.search(trecho)) if familia == "sumula" else False,
        lei_numerada=lei_numerada,
        identificador_suficiente=tem_identificador(trecho),
        identificador_adulterado=identificador_adulterado,
        fonte=candidato.get("fonte", "regra"),
    )
