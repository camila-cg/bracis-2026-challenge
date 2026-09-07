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

# --------------------------------------------------------------------------- #
# Localização dos arquivos
# --------------------------------------------------------------------------- #

_MARCADOR = Path("data") / "goldenset.csv"


def raiz_projeto(inicio: Path | str | None = None) -> Path:
    """Sobe a partir de `inicio` até achar a pasta que contém data/goldenset.csv.

    Permite rodar os notebooks tanto de src/ quanto da raiz do repositório.
    """
    atual = Path(inicio or Path.cwd()).resolve()
    for candidato in [atual, *atual.parents]:
        if (candidato / _MARCADOR).exists():
            return candidato
    raise FileNotFoundError(
        "Não encontrei data/goldenset.csv subindo a partir de "
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
    """O goldenset.csv grava quebra de linha como a sequência literal \\n."""
    return trecho.replace(_BARRA_N, "\n")


def carregar_gabarito() -> list[dict]:
    """Linhas do goldenset.csv, com inicio/fim como int e trecho desescapado.

    Acrescenta a chave 'trecho_bruto' com o valor original do arquivo.
    """
    with open(caminho("data", "goldenset.csv"), encoding="utf-8", newline="") as fh:
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
CNJ = re.compile(r"\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")

# Número no estilo STF/STJ: 1.741.784 / 1741784 / 1.741. 784
NUMERO_LIVRE = re.compile(r"\d[\d.\s]{2,}\d|\d{3,}")

# Confusões de OCR que aparecem no nível 2. A garantia do desafio é que um
# dígito nunca vira outro dígito — mas uma LETRA pode ocupar o lugar de um
# dígito (21737l8, 1.528.4S5, 1.45g.779, 170076O). O mapa só deve ser aplicado
# DENTRO do token numérico: aplicá-lo ao trecho inteiro destrói a sigla da
# classe processual ("RESP" viraria "RE5P").
OCR_PARA_DIGITO = {
    "l": "1", "I": "1", "i": "1", "|": "1",
    "O": "0", "o": "0", "D": "0",
    "S": "5", "s": "5",
    "g": "9", "q": "9",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6", "b": "6",
    "T": "7",
}

_TOKEN_NUMERICO = re.compile(r"[0-9][0-9OoSslIigqBZzGbTD.\s-]*[0-9OoSslIigqBZzGbTD]")


def corrigir_ocr_em_numeros(texto: str) -> str:
    """Troca letra por dígito apenas dentro de tokens que já parecem número."""

    def _troca(m: re.Match) -> str:
        return "".join(OCR_PARA_DIGITO.get(c, c) for c in m.group(0))

    return _TOKEN_NUMERICO.sub(_troca, texto)


def agrupar_milhares(digitos: str) -> str:
    """1741784 -> 1.741.784 (o formato que o tokenizador unicode61 indexa)."""
    partes, resto = [], digitos
    while len(resto) > 3:
        partes.insert(0, resto[-3:])
        resto = resto[:-3]
    partes.insert(0, resto)
    return ".".join(partes)


def variantes_numero(trecho: str) -> list[str]:
    """Formas do identificador que vale tentar no índice FTS, em ordem de aposta.

    Cobre: padrão CNJ (com e sem zeros à esquerda), número pontuado de três em
    três, e a sequência crua de dígitos. Aplica a correção de OCR antes.
    """
    limpo = corrigir_ocr_em_numeros(trecho)
    sem_espaco = re.sub(r"\s+", "", limpo)

    m = CNJ.search(sem_espaco)
    if m:
        achado = m.group(0)
        formas = [achado]
        sequencial, resto = achado.split("-", 1)
        if len(sequencial) < 7:
            formas.append(sequencial.zfill(7) + "-" + resto)
        return formas

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
