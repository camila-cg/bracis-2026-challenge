"""
Utilitários compartilhados pelos notebooks de EDA do desafio BRACIS 2026 / Jusbrasil.

Nada aqui depende de pandas ou matplotlib: são funções de carga e normalização
que também servem de ponto de partida para o pipeline de solução.

Uso típico dentro de um notebook:

    import eda_utils as u
    docs = u.carregar_documentos()
    gab  = u.carregar_gabarito()
    con  = u.conectar_base()
"""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from urllib.parse import quote

from configuracao import carregar_config

# --------------------------------------------------------------------------- #
# Localização dos arquivos
# --------------------------------------------------------------------------- #

_MARCADOR = Path("data") / "goldenset_offsets.csv"


def raiz_projeto(inicio: Path | str | None = None) -> Path:
    """Sobe a partir de `inicio` até achar a pasta que contém data/goldenset_offsets.csv.

    Permite rodar os notebooks tanto de notebooks/ quanto da raiz do repositório.
    """
    atual = Path(inicio or Path.cwd()).resolve()
    for candidato in [atual, *atual.parents]:
        if (candidato / _MARCADOR).exists():
            return candidato
    raise FileNotFoundError(
        "Não encontrei data/goldenset_offsets.csv subindo a partir de "
        f"{atual}. Rode o notebook de dentro do repositório."
    )


def caminho(*partes: str) -> Path:
    """Caminho absoluto a partir da raiz do projeto: caminho('data', 'txt')."""
    return raiz_projeto().joinpath(*partes)


# --------------------------------------------------------------------------- #
# Documentos de entrada
# --------------------------------------------------------------------------- #


def nivel(documento_id: str) -> int:
    """1 ou 2, lido do nome do arquivo (gen_n1_007 -> 1)."""
    return int(documento_id.split("_")[1][1:])


def carregar_documentos() -> dict[str, str]:
    """{documento_id: texto}. Lê como UTF-8 e preserva os offsets originais.

    Os .txt do desafio usam LF puro, então a leitura padrão do Python não
    desloca nenhum offset. Se algum dia vierem com CRLF, a leitura universal
    colapsaria \\r\\n em \\n e quebraria a contagem de codepoints — por isso a
    checagem explícita abaixo.
    """
    docs = {}
    for arq in sorted(caminho("data", "txt").glob("*.txt")):
        bruto = arq.read_bytes()
        if b"\r\n" in bruto:
            raise ValueError(
                f"{arq.name} contém CRLF; leia com newline='' para não deslocar offsets."
            )
        docs[arq.stem] = bruto.decode("utf-8")
    return docs


# --------------------------------------------------------------------------- #
# Gabarito
# --------------------------------------------------------------------------- #

_BARRA_N = "\\" + "n"


def desescapar(trecho: str) -> str:
    """O goldenset_offsets.csv grava quebra de linha como a sequência literal \\n."""
    return trecho.replace(_BARRA_N, "\n")


def carregar_gabarito() -> list[dict]:
    """Linhas do goldenset_offsets.csv, com inicio/fim como int e trecho desescapado.

    Acrescenta a chave 'trecho_bruto' com o valor original do arquivo.
    """
    # utf-8-sig: o arquivo enviado na correção do dataset vem com BOM: abrir
    # como "utf-8" comum deixaria o BOM grudado no nome da 1a coluna
    # ("﻿nivel" em vez de "nivel") e quebraria o DictReader inteiro.
    with open(caminho("data", "goldenset_offsets.csv"), encoding="utf-8-sig", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    for linha in linhas:
        linha["inicio"] = int(linha["inicio"])
        linha["fim"] = int(linha["fim"])
        linha["nivel"] = int(linha["nivel"])
        linha["trecho_bruto"] = linha["trecho"]
        linha["trecho"] = desescapar(linha["trecho"])
    return linhas


# --------------------------------------------------------------------------- #
# Base canônica
# --------------------------------------------------------------------------- #


def _uri_sqlite(arquivo: Path) -> str:
    """URI file: aceita pelo SQLite, inclusive para caminho de rede (UNC).

    Um caminho UNC vira //servidor/share/... em POSIX; concatenar isso direto
    em "file:" faz o SQLite ler "servidor" como autoridade da URI e recusar.
    A forma que funciona é file:////servidor/share/...
    """
    posix = quote(arquivo.as_posix(), safe="/:")
    if posix.startswith("//"):
        return "file://" + posix
    return "file:///" + posix.lstrip("/")


def conectar_base(somente_leitura: bool = True) -> sqlite3.Connection:
    """Conexão com desafio1_bracis.db.

    O modo `immutable=1` é o que faz a leitura funcionar quando o arquivo está
    num compartilhamento de rede (WSL/UNC), onde o SQLite não consegue criar o
    lock e devolve "database is locked". Ele também informa ao SQLite que o
    arquivo não muda durante a sessão, o que dispensa qualquer escrita.
    """
    arquivo = caminho("data", "desafio1_bracis.db")
    if somente_leitura:
        return sqlite3.connect(_uri_sqlite(arquivo) + "?immutable=1", uri=True)
    return sqlite3.connect(str(arquivo))


def consulta(con: sqlite3.Connection, sql: str, params=()) -> list[tuple]:
    return con.execute(sql, params).fetchall()


_conexao_cache: sqlite3.Connection | None = None


def conexao_padrao() -> sqlite3.Connection:
    """Conexão única reaproveitada pelos helpers deste módulo."""
    global _conexao_cache
    if _conexao_cache is None:
        _conexao_cache = conectar_base()
    return _conexao_cache


# --------------------------------------------------------------------------- #
# Normalização de identificadores
# --------------------------------------------------------------------------- #

# Número no padrão CNJ. A parte sequencial tem 7 dígitos no padrão oficial, mas
# TST e TSE frequentemente a escrevem sem os zeros à esquerda (ARR-213-85.2010...).
#
# As fronteiras de dígito são parte do contrato: sem elas, o regex encontrava
# um CNJ válido dentro de um identificador adulterado, por exemplo
# ``97000136-79.2023.7.00.0000`` -> ``7000136-79.2023.7.00.0000``.
CNJ = re.compile(
    r"(?<!\d)(?<!\d[./–—-])(?P<sequencial>\d{1,7})-"
    r"(?P<dv>\d{2})\.(?P<ano>\d{4})\.(?P<segmento>\d)\."
    r"(?P<tribunal>\d{2})\.(?P<origem>\d{4})(?!\d)(?![./–—-]\s*\d)"
)

_SEPARADOR_NUMERICO = r"[./–—-]"

# Forma usada como chave no índice de acórdãos: sequencial oficial com sete
# dígitos, hífen e os 13 dígitos estruturados restantes sem pontuação.
_CHAVE_CNJ = re.compile(
    r"(?<!\d)(?P<sequencial>\d{7})-(?P<fim>\d{13})(?!\d)"
)

# Se há o início inequívoco de um CNJ, mas o padrão estrito não fecha, não
# devemos reinterpretar apenas o sequencial como um número livre. Esse bloqueio
# evita que um identificador malformado escape para o fallback genérico.
_CANDIDATO_CNJ = re.compile(r"(?<!\d)\d+-\d{2}\.")

# Cauda de um CNJ: DV.ano.segmento.tribunal.origem, sem depender do que vem antes.
_CAUDA_CNJ_PADRAO = r"\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}"
_CAUDA_CNJ = re.compile(rf"(?<!\d){_CAUDA_CNJ_PADRAO}")

# Número no estilo STF/STJ: 1.741.784 / 1741784 / 1.741. 784
NUMERO_LIVRE = re.compile(r"\d[\d.\s]{2,}\d|\d{3,}")

# Confusões de OCR que aparecem no nível 2. A garantia do desafio é que um
# dígito nunca vira outro dígito — mas uma LETRA pode ocupar o lugar de um
# dígito (21737l8, 1.528.4S5, 1.45g.779, 170076O). O mapa só deve ser aplicado
# DENTRO do token numérico: aplicá-lo ao trecho inteiro destrói a sigla da
# classe processual ("RESP" viraria "RE5P").
OCR_PARA_DIGITO = carregar_config("regras_ocr")["letra_para_digito"]

_LETRAS_OCR = re.escape("".join(OCR_PARA_DIGITO))
_CARACTERES_NUMERICOS_OCR = "0-9" + _LETRAS_OCR
_CNJ_COM_OCR = re.compile(
    rf"(?<![A-Za-z0-9])[{_CARACTERES_NUMERICOS_OCR}]{{1,7}}-"
    rf"[{_CARACTERES_NUMERICOS_OCR}]{{2}}\."
    rf"[{_CARACTERES_NUMERICOS_OCR}]{{4}}\."
    rf"[{_CARACTERES_NUMERICOS_OCR}]\."
    rf"[{_CARACTERES_NUMERICOS_OCR}]{{2}}\."
    rf"[{_CARACTERES_NUMERICOS_OCR}]{{4}}(?![A-Za-z0-9])"
)
_LETRA_ADJACENTE_A_DIGITO = re.compile(
    rf"\d[.\s/–—-]*[{_LETRAS_OCR}]+(?=[.\s/–—-]*\d)|"
    rf"(?<=\d)[{_LETRAS_OCR}]{{2,}}(?![A-Za-z])|"
    rf"(?<=\d\.)[{_LETRAS_OCR}]{{2,}}(?=(?:[.\s\-]*\d|$))|"
    rf"(?<![A-Za-z])[{_LETRAS_OCR}]{{1,2}}(?=\d)|"
    rf"(?<=\d)[{_LETRAS_OCR}](?![A-Za-z])|"
    rf"(?<=\d\.)[{_LETRAS_OCR}](?=\d)"
)


# OCR e editores trocam o hífen do CNJ por outros traços: hífen Unicode, hífen
# não separável, traço numérico, meia-risca, travessão, barra horizontal e sinal
# de menos. O traço só é tratado como hífen quando a cauda do CNJ
# (DV.ano.J.TR.origem) vem logo depois; intervalos ("12—15") ficam como estão.
_TRACOS_TIPOGRAFICOS = "‐-―−"
# Colado aos dois lados: troca um caractere por outro, o comprimento não muda.
_TRACO_COLADO_A_CAUDA_CNJ = re.compile(rf"(?<=\d)[{_TRACOS_TIPOGRAFICOS}](?={_CAUDA_CNJ_PADRAO})")
# Com espaços em volta: muda o comprimento, só serve onde não há posições a preservar.
_TRACO_ANTES_DA_CAUDA_CNJ = re.compile(rf"(?<=\d)\s*[{_TRACOS_TIPOGRAFICOS}]\s*(?={_CAUDA_CNJ_PADRAO})")


def normalizar_traco_de_cnj(texto: str) -> str:
    """Traço tipográfico (com ou sem espaços) antes da cauda de um CNJ vira hífen."""
    return _TRACO_ANTES_DA_CAUDA_CNJ.sub("-", texto)


def corrigir_ocr_em_numeros(texto: str) -> str:
    """Corrige OCR numérico sem atravessar prefixos, classes ou tribunais.

    Preserva o comprimento do texto: há quem use posições do resultado.
    """

    def _trocar(m: re.Match) -> str:
        return "".join(OCR_PARA_DIGITO.get(c, c) for c in m.group())

    texto = _TRACO_COLADO_A_CAUDA_CNJ.sub("-", texto)
    # A estrutura completa permite corrigir segmentos CNJ compostos apenas por
    # letras OCR (``.OO.OOOO``) sem transformar prefixos como ``TST-123``.
    texto = _CNJ_COM_OCR.sub(_trocar, texto)
    anterior = None
    while anterior != texto:
        anterior = texto
        texto = _LETRA_ADJACENTE_A_DIGITO.sub(_trocar, texto)
    return texto


def agrupar_milhares(digitos: str) -> str:
    """1741784 -> 1.741.784 (o formato que o tokenizador unicode61 indexa)."""
    partes, resto = [], digitos
    while len(resto) > 3:
        partes.insert(0, resto[-3:])
        resto = resto[:-3]
    partes.insert(0, resto)
    return ".".join(partes)


def _buscar_cnj_isolado(texto: str) -> re.Match | None:
    """Encontra CNJ sem aceitar componentes numéricos separados por sinais."""
    for achado in CNJ.finditer(texto):
        esquerda = texto[:achado.start()]
        direita = texto[achado.end():]
        embutido_a_esquerda = re.search(
            rf"\d(?:\s*{_SEPARADOR_NUMERICO}\s*)+$", esquerda
        )
        embutido_a_direita = re.match(
            rf"^(?:\s*{_SEPARADOR_NUMERICO}\s*)+\d", direita
        )
        if not embutido_a_esquerda and not embutido_a_direita:
            return achado
    return None


def _componentes_cnj_exato(numero: str) -> tuple[str, str] | None:
    """Lê um CNJ isolado sem aceitar sobras numéricas ou textuais.

    O retorno é ``(sequencial, fim)``; ``fim`` contém os 13 dígitos depois do
    hífen. Aceita tanto a grafia CNJ quanto a chave interna do índice e corrige
    ruído OCR reversível antes da validação.
    """
    limpo = re.sub(r"\s+", "", corrigir_ocr_em_numeros(numero))
    m = CNJ.fullmatch(limpo)
    if m:
        fim = "".join(
            m.group(campo) for campo in ("dv", "ano", "segmento", "tribunal", "origem")
        )
        return m.group("sequencial"), fim

    m = _CHAVE_CNJ.fullmatch(limpo)
    if m:
        return m.group("sequencial"), m.group("fim")
    return None


def chave_cnj(numero: str) -> str | None:
    """Converte um CNJ isolado para a chave exata usada no índice.

    Retorna ``None`` se houver mais de sete dígitos no sequencial, dígito
    adjacente, estrutura incompleta ou qualquer conteúdo além do número.
    """
    componentes = _componentes_cnj_exato(numero)
    if componentes is None:
        return None
    sequencial, fim = componentes
    return f"{sequencial.zfill(7)}-{fim}"


def formatar_cnj_para_fts(numero: str) -> str | None:
    """Formata uma chave CNJ isolada para uma consulta exata no FTS.

    A função é deliberadamente estrita para que o fallback receba somente o
    identificador já selecionado, nunca outro número incidental do span.
    """
    chave = chave_cnj(numero)
    if chave is None:
        return None
    sequencial, fim = chave.split("-", 1)
    return f"{sequencial}-{fim[:2]}.{fim[2:6]}.{fim[6]}.{fim[7:9]}.{fim[9:]}"


def variantes_numero(trecho: str) -> list[str]:
    """Formas do identificador que vale tentar no índice FTS, em ordem de aposta.

    Cobre: padrão CNJ (com e sem zeros à esquerda), número pontuado de três em
    três, e a sequência crua de dígitos. Aplica a correção de OCR antes.
    """
    limpo = corrigir_ocr_em_numeros(trecho)
    # Sem espaços, um traço que tinha espaço em volta também fica colado.
    sem_espaco = _TRACO_COLADO_A_CAUDA_CNJ.sub("-", re.sub(r"\s+", "", limpo))

    # Permite que o resolvedor receba diretamente a chave já selecionada por
    # ``normalizar_citacao``. Essa é a entrada segura para o fallback FTS.
    forma_fts = formatar_cnj_para_fts(sem_espaco)
    if forma_fts is not None:
        m_exato = CNJ.fullmatch(sem_espaco)
        if m_exato and len(m_exato.group("sequencial")) < 7:
            return [m_exato.group(0), forma_fts]
        # A normalização interna completa o sequencial com zeros. Preserve
        # também a grafia curta usada nos cabeçalhos de TST/TSE quando o
        # fallback recebe apenas essa chave canônica.
        m_chave = _CHAVE_CNJ.fullmatch(sem_espaco)
        if m_chave:
            sequencial_curto = m_chave.group("sequencial").lstrip("0") or "0"
            if len(sequencial_curto) < 7:
                fim = m_chave.group("fim")
                forma_curta = (
                    f"{sequencial_curto}-{fim[:2]}.{fim[2:6]}."
                    f"{fim[6]}.{fim[7:9]}.{fim[9:]}"
                )
                return [forma_fts, forma_curta]
        return [forma_fts]

    m = _buscar_cnj_isolado(sem_espaco)
    if m:
        achado = m.group(0)
        formas = [achado]
        sequencial = m.group("sequencial")
        if len(sequencial) < 7:
            formas.append(formatar_cnj_para_fts(achado))
        return formas

    # Um CNJ reconhecivelmente iniciado, porém inválido, não pode cair para a
    # heurística de número livre: isso recriaria uma correspondência parcial.
    if _CANDIDATO_CNJ.search(sem_espaco):
        return []

    # A cauda de um CNJ (DV.ano.J.TR.origem) sem um CNJ válido é um CNJ partido
    # (separador estranho entre o sequencial e o DV). O que sobra antes dela, o
    # sequencial, não identifica nada: mandá-lo ao FTS casaria com qualquer
    # documento que mencione aquele número.
    if _CAUDA_CNJ.search(sem_espaco):
        return []

    m = NUMERO_LIVRE.search(re.sub(r"\s+", " ", limpo))
    if not m:
        return []
    digitos = re.sub(r"[^0-9]", "", m.group(0))
    if not digitos:
        return []
    formas = [agrupar_milhares(digitos)]
    if digitos not in formas:
        formas.append(digitos)
    return formas


# Estrutura de um CNJ sem exigir fronteiras: serve só para reconhecer que o
# trecho *tenta* ser um CNJ, com um ou dois símbolos entre o sequencial e o DV.
# Mais estreita que ``_CANDIDATO_CNJ`` de propósito, para que um intervalo de
# páginas ("fls. 12-14.") não conte como CNJ.
_ESTRUTURA_CNJ = re.compile(r"\d+[^\w\s.]{1,2}\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")


def cnj_adulterado(trecho: str) -> bool:
    """O trecho traz a estrutura de um CNJ, mas nenhum CNJ isolado e válido.

    Lê o trecho como ``variantes_numero`` (OCR corrigido, sem espaços), então os
    dois concordam: quando ``variantes_numero`` descarta o bloco por ter dígito
    a mais, sufixo, prefixo ou separador estranho, esta função diz que foi
    adulteração. Um identificador explícito e inválido não é identificador
    ausente.
    """
    sem_espaco = _TRACO_COLADO_A_CAUDA_CNJ.sub("-", re.sub(r"\s+", "", corrigir_ocr_em_numeros(trecho)))
    if formatar_cnj_para_fts(sem_espaco) is not None or _buscar_cnj_isolado(sem_espaco):
        return False
    return bool(_ESTRUTURA_CNJ.search(sem_espaco))


# Janela, por tribunal, dentro da qual o número do processo aparece no documento
# que É aquele processo. Medida em 02_eda_base_canonica.ipynb — ver a seção
# "Onde mora o identificador".
JANELA_CABECALHO = {
    "STM": 300,
    "TSE": 300,
    "STF": 300,
    "STJ": 300,
    "TST": 2500,   # o TST só cita o número na fórmula "Vistos, relatados e discutidos"
    None: 300,
}


def resolver_por_numero(trecho: str, janela_padrao: int = 2500,
                        con: sqlite3.Connection | None = None) -> dict[int, int]:
    """Candidatos {id: offset} cujo cabeçalho contém o número citado.

    Baseline honesto do desafio: busca por frase no FTS e descarta os documentos
    que apenas MENCIONAM o número no corpo. Ver 04_analise_integrada.ipynb.

    A janela usada é a de `JANELA_CABECALHO` para o tribunal do candidato, e
    `janela_padrao` quando o tribunal não está no dicionário.
    """
    con = con or conexao_padrao()
    candidatos: dict[int, int] = {}
    for forma in variantes_numero(trecho):
        try:
            achados = con.execute(
                "SELECT d.id, d.tribunal, d.texto FROM documentos_fts f "
                "JOIN documentos d ON d.rowid = f.rowid "
                "WHERE documentos_fts MATCH ?",
                ('"' + forma + '"',),
            ).fetchall()
        except sqlite3.OperationalError:
            continue  # sintaxe rejeitada pelo FTS5
        for doc_id, tribunal, texto in achados:
            pos = texto.find(forma)
            limite = JANELA_CABECALHO.get(tribunal, janela_padrao)
            if 0 <= pos < limite:
                candidatos[doc_id] = min(candidatos.get(doc_id, 10 ** 9), pos)
    return candidatos
