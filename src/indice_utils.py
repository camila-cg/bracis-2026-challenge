"""
Índice da base canônica — Desafio 1, BRACIS 2026 / Jusbrasil.

Complementa eda_utils.py: parsing de cabeçalho por tribunal e montagem do
índice (tribunal, classe, número, origem) -> id, pra resolver uma citação
por lookup direto em vez de varrer o FTS.

Uso:

    import eda_utils as u
    import indice_utils as ind
    con = u.conectar_base()
    acordaos, sem_parser = ind.construir_indice_acordaos(con)
    normativos, sem_resolucao = ind.construir_indice_normativos(con)
    ind.exportar_json(acordaos, normativos, "data/indice_canonico.json")
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone

import eda_utils as u
from configuracao import carregar_config

# --------------------------------------------------------------------------- #
# Parsers de cabeçalho por tribunal
# --------------------------------------------------------------------------- #

# A classe pega a cadeia processual inteira, não só a última palavra: "EMB.DIV.
# NOS EMB.DECL. NO AG.REG. NO RE" e "... NO SEGUNDO AG.REG. NO RE" são decisões
# diferentes do mesmo processo, cada uma com seu id. Cortando a cadeia as duas
# colidem na mesma chave.
STF_CABECALHO = re.compile(
    r"\d{2}/\d{2}/\d{4}\s+"
    r"(?P<classe>[A-ZÇÃÕÁÉÍÓÚÂÊÔ][A-ZÇÃÕÁÉÍÓÚÂÊÔ0-9.\s]+?)\s+"
    r"(?P<numero>\d{1,3}(?:\.\d{3})*)\s+"
    r"(?P<uf>[A-ZÀ-Ü][A-ZÀ-Üa-z\s]+?)\s+RELATOR",
    re.UNICODE,
)

# O parênteses depois da UF é o protocolo (ano/sequência) do STJ. O "Nº"
# principal se repete entre processos diferentes; o protocolo não. Entra na
# chave junto com a classe.
STJ_CABECALHO = re.compile(
    r"(?P<classe>(?:[A-Za-z]+\s+no\s+)*[A-ZÇÃÕÁÉÍÓÚÂÊÔ][A-ZÇÃÕÁÉÍÓÚÂÊÔ\s]+?)\s+"
    r"N[ºo°]\s*(?P<numero>[\d.]+)\s*-\s*(?P<uf>[A-Z]{2})\s*"
    r"\((?P<origem>[^)]+)\)\s*RELATOR",
    re.UNICODE,
)

# Dois preâmbulos diferentes ("Poder Judiciário STM EXTRATO DE ATA..." e
# "Secretaria do Tribunal Pleno..."), mas o miolo é igual nos dois, por isso
# o regex não âncora no início.
STM_CABECALHO = re.compile(
    r"(?P<classe>[A-ZÇÃÕÁÉÍÓÚÂÊÔ][A-ZÇÃÕÁÉÍÓÚÂÊÔ\s]+?)\s+"
    r"N[ºo°]\s*(?P<numero>\d{7}-?\s*\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})"
    r"(?:/(?P<uf>[A-Z]{2}))?",
    re.UNICODE,
)

# TSE é o pior caso: pontuação inconsistente, "-" e "—" misturados, marcador
# "Nº" às vezes vira "N o" (com espaço) ou some, número às vezes vem separado
# só por espaço em vez de ponto, e "Relator" pode vir com ruído de OCR
# (l -> I). Ainda sobra uma fração com corrupção de encoding de verdade
# (letra acentuada virou símbolo, não é problema de exibição) — essa não
# tem conserto por regex, fica de fora do índice.
TSE_CABECALHO = re.compile(
    r"(?P<classe>[A-ZÇÃÕÁÉÍÓÚÂÊÔ][A-ZÇÃÕÁÉÍÓÚÂÊÔ\s]+?)\s+"
    r"(?:N\s?[º°o]?\.?\s*)?(?P<numero>\d[\d.\s\-]*\d)\s*(?:\(\s*[\d.\-]+\s*\))?\s*"
    r"[-—]?\s*(?:CLASSE\s*\d+[ªa]?\s*\d*[-—]?\s*)?[-—]?\s*"
    r"(?P<uf>[A-ZÀ-Üa-zà-ü.\s\-]+?)\s+R\s*e\s*[lI1]\s*a\s*t\s*o",
    re.UNICODE,
)

# Alguns cabeçalhos do TSE têm o encoding quebrado (Ó, º e Ç viram €, ‚ e ƒ), o
# que impede TSE_CABECALHO. O CNJ completo continua legível; a classe só entra
# se estiver em maiúsculas puras e o número fecha o CNJ (com DV) até a "CLASSE".
# A UF só vale quando o nome do estado está legível; com acento quebrado fica None.
TSE_CABECALHO_ENCODING_QUEBRADO = re.compile(
    r"TRIBUNAL\s+SUPERIOR\s+ELEITORAL\s+AC\S{1,2}RD\s?\S{0,2}\s+"
    r"(?P<classe>[A-Z][A-Z\s]+?)\s+N\S?\s*"
    r"(?P<numero>\d{1,7}-\d{2}\.\s?\d{4}\.\d\.\d{2}\.\d{4})\s*-?\s*CLASSE\s*\d+\S?"
    r"(?:\s+[^-]*?-\s*(?P<uf>[A-Za-z\s]+?)\s+Relator)?",
)

# Uma fração dos acórdãos do TSE vem como página de andamento processual
# ("Número: ... Classe: ... Órgão julgador...") em vez do cabeçalho de
# acórdão de verdade — layout completamente diferente, regex à parte.
TSE_ANDAMENTO = re.compile(
    r"N[uú]mero:\s*(?P<numero>[\d.\-]+)\s*Classe:\s*"
    r"(?P<classe>[A-ZÇÃÕÁÉÍÓÚÂÊÔ][A-ZÇÃÕÁÉÍÓÚÂÊÔ\s]+?)\s+[ÓO]rg[ãa]o",
    re.UNICODE,
)

# TST não tem cabeçalho: abre na ementa, e o número só aparece na fórmula
# "Vistos, relatados e discutidos estes autos de <classe> nº TST-<código>-
# <número CNJ>, em que é...". Às vezes vem com espaço em volta dos hifens.
TST_FORMULA = re.compile(
    r"n\s*\.?\s*(?:[ºo°]|[^\sA-Za-z0-9])?\s*(?P<raw>TST[\s\-A-Za-z0-9./]{5,60}?\d{2}\.\d{4}\.5\.\d{2}\.\d{4})",
    re.UNICODE | re.DOTALL,
)
TST_NUMERO = re.compile(r"^TST-(?P<codigo>[A-Za-z-]+)-(?P<numero>\d+-\d{2}\.\d{4}\.5\.\d{2}\.\d{4})$")

# O extrato de ata do STM pode empurrar o título do processo para além dos 500
# caracteres do cabeçalho, e o sequencial dos processos antigos é curto
# ("APELAÇÃO Nº 234-73.2016.7.11.0211/DF"). Só vale se o mesmo número repete no
# documento (cabeçalho de página), o que o distingue de um processo apenas citado.
STM_CABECALHO_CURTO = re.compile(
    r"(?P<classe>[A-ZÇÃÕÁÉÍÓÚÂÊÔ][A-ZÇÃÕÁÉÍÓÚÂÊÔ]+)\s+"
    r"N[ºo°]\s*(?P<numero>\d{1,7}-\d{2}\.\d{4}\.7\.\d{2}\.\d{4})"
    r"(?:\s*[-/]\s*(?P<uf>[A-Z]{2})\b)?",
    re.UNICODE,
)
STM_JANELA_CURTO = 3000
STM_REPETICOES_MINIMAS = 3

PARSERS = {
    "STF": [STF_CABECALHO],
    "STJ": [STJ_CABECALHO],
    "STM": [STM_CABECALHO],
    "TSE": [TSE_CABECALHO, TSE_CABECALHO_ENCODING_QUEBRADO, TSE_ANDAMENTO],
}

UF_POR_EXTENSO = carregar_config("tribunais")["uf_por_extenso"]


def _sem_acento(s: str) -> str:
    troca = str.maketrans("ÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ", "AAAAAEEEEIIIIOOOOOUUUUC")
    return s.translate(troca)


def normalizar_uf(bruto: str | None) -> str | None:
    if not bruto:
        return None
    limpo = re.sub(r"[.\-\s]+$", "", bruto.strip().upper())
    limpo = re.sub(r"^[.\-\s]+", "", limpo)
    if re.fullmatch(r"[A-Z]{2}", limpo):
        return limpo
    return UF_POR_EXTENSO.get(_sem_acento(limpo))


def normalizar_numero(bruto: str) -> str:
    """Só os dígitos, forma canônica igual para o mesmo processo.

    Pra CNJ, completo a parte sequencial com zero à esquerda quando falta
    (TST/TSE às vezes escrevem sem os 7 dígitos oficiais). Fora do CNJ, um
    dígito nunca vira outro dígito, então reduzir a dígitos crus já basta.
    """
    cnj = u.chave_cnj(bruto)
    if cnj is not None:
        return cnj
    return re.sub(r"[^0-9]", "", bruto)


def _cnj_por_digitos_sem_separadores(digitos: str) -> str | None:
    """Reconstrói um CNJ apenas quando a cauda de 13 dígitos é plausível."""
    if not 14 <= len(digitos) <= 20:
        return None
    sequencial, fim = digitos[:-13], digitos[-13:]
    ano = int(fim[2:6])
    segmento = fim[6]
    if not 1900 <= ano <= 2099 or segmento == "0":
        return None
    return f"{sequencial.zfill(7)}-{fim}"


def _recuperar_cnj_tse(bruto: str, texto: str) -> str | None:
    """Recupera CNJ danificado no cabeçalho quando o corpo confirma a identidade.

    O parser do TSE às vezes captura um CNJ que perdeu pontos ou o DV. A função
    não corrige só pela aparência: exige que a forma reconstruída reapareça como
    CNJ válido no documento. Se a cauda perdeu o DV, aceita a repetição única
    que tenha o mesmo sequencial escrito antes do hífen no cabeçalho.
    """
    if u.chave_cnj(bruto) is not None:
        return None
    digitos = re.sub(r"\D", "", bruto)
    sequencial_bruto = re.match(r"^\s*(\d{1,7})\s*[-–—]", bruto)
    if not sequencial_bruto or not 14 <= len(digitos) <= 20:
        return None

    texto_normalizado = u.normalizar_traco_de_cnj(
        u.corrigir_ocr_em_numeros(texto)
    )
    encontrados = {
        chave
        for achado in u.CNJ.finditer(texto_normalizado)
        if (chave := u.chave_cnj(achado.group())) is not None
    }
    sequencial = sequencial_bruto.group(1).lstrip("0") or "0"
    candidato = _cnj_por_digitos_sem_separadores(digitos)
    sequencial_candidato = (
        candidato[:7].lstrip("0") or "0"
        if candidato is not None
        else None
    )
    if candidato in encontrados and sequencial_candidato == sequencial:
        return candidato

    mesmo_sequencial = {
        chave
        for chave in encontrados
        if (chave[:7].lstrip("0") or "0") == sequencial
    }
    if len(mesmo_sequencial) == 1:
        return next(iter(mesmo_sequencial))
    return None


# --------------------------------------------------------------------------- #
# Construção do índice de acórdãos
# --------------------------------------------------------------------------- #


def parsear_cabecalho(tribunal: str, texto: str) -> dict | None:
    """{"classe", "numero", "uf", "origem"} a partir do cabeçalho, ou None."""
    if tribunal == "TST":
        m = TST_FORMULA.search(texto)
        if not m:
            return None
        bruto = re.sub(r"\s+", "", m.group("raw"))
        mf = TST_NUMERO.match(bruto)
        if not mf:
            return None
        return {"classe": mf.group("codigo"), "numero": mf.group("numero"), "uf": None, "origem": None}

    padroes = PARSERS.get(tribunal)
    if padroes is None:
        return None
    janela = texto[:500] if tribunal != "STF" else texto[:400]
    for padrao in padroes:
        m = padrao.search(janela)
        if m:
            return {
                "classe": re.sub(r"\s+", " ", m.group("classe")).strip(),
                "numero": m.group("numero"),
                "uf": m.group("uf") if "uf" in padrao.groupindex else None,
                "origem": m.group("origem") if "origem" in padrao.groupindex else None,
            }
    if tribunal == "STM":
        m = STM_CABECALHO_CURTO.search(texto[:STM_JANELA_CURTO])
        if m and texto.count(m.group("numero")) >= STM_REPETICOES_MINIMAS:
            return {
                "classe": m.group("classe"),
                "numero": m.group("numero"),
                "uf": m.group("uf"),
                "origem": None,
            }
    return None


def normalizar_classe(bruto: str) -> str:
    return re.sub(r"\s+", " ", bruto.upper().strip())


def construir_indice_acordaos(con: sqlite3.Connection) -> tuple[dict, list[dict]]:
    """Índice (tribunal, classe, número, origem) -> [registros] dos acórdãos.

    A chave usa a classe inteira porque número sozinho não é único: um
    processo pode ter vários acórdãos (decisão, agravo sobre a decisão,
    embargos sobre o agravo), cada um com id próprio.

    Devolve o índice e a lista de registros que o parser não leu — nunca
    adivinho, fica de fora e listado. Colisão residual (mesma chave, ids
    diferentes) é duplicata do acervo; guardo a lista inteira em vez de
    escolher um id às cegas.
    """
    indice: dict[tuple, list[dict]] = {}
    sem_parser = []
    cur = con.execute(
        "SELECT documento_id, id, tribunal, texto, texto_len FROM documentos WHERE natureza='acordao'"
    )
    for documento_id, id_canonico, tribunal, texto, texto_len in cur:
        campos = parsear_cabecalho(tribunal, texto)
        if campos is None:
            sem_parser.append({"documento_id": documento_id, "id": id_canonico, "tribunal": tribunal})
            continue
        numero_norm = normalizar_numero(campos["numero"])
        if tribunal == "TSE":
            numero_norm = _recuperar_cnj_tse(campos["numero"], texto) or numero_norm
        classe_norm = normalizar_classe(campos["classe"])
        origem_norm = re.sub(r"[^0-9]", "", campos["origem"]) if campos["origem"] else None
        chave = (tribunal, classe_norm, numero_norm, origem_norm)
        indice.setdefault(chave, []).append({
            "id": id_canonico,
            "documento_id": documento_id,
            "classe": campos["classe"],
            "uf": normalizar_uf(campos["uf"]),
            "texto_len": texto_len,
        })
    return indice, sem_parser


# --------------------------------------------------------------------------- #
# Índice dos registros normativos (súmulas, temas + dispositivos de lei)
# --------------------------------------------------------------------------- #

# Estatuto de cada dispositivo mapeado à mão (o número de artigo sozinho não
# basta, se repete entre códigos diferentes).
ESTATUTO_POR_ID = {int(chave): valor for chave, valor in carregar_config("diplomas")["estatuto_por_id"].items()}

# O texto dos dispositivos abre com "Artigo 276 da Lei nº ..."; sem o "(?:igo)?"
# o regex parava em "Art" + "igo" e nenhum dos 13 dispositivos entrava no índice.
ARTIGO_INICIAL = re.compile(r"\s*Art(?:igo)?\.?\s*(\d[\d.]*)", re.I)

# O título preservado no início do texto é a fonte primária da identidade.
# Aceita as formas usadas pelas bases canônicas, como "Súmula n. 331" e
# "Súmula Vinculante nº 10", sem depender do id canônico do registro.
SUMULA_NO_TITULO = re.compile(
    r"\A\s*S[ÚU]MULA(?P<vinculante>\s+VINCULANTE)?\s*"
    r"(?:N(?:(?:[ÚU]MERO)|(?:\s*\.?\s*[º°O]?))\s*)?"
    r"(?P<numero>\d+(?:\.\d+)*)\b",
    re.I | re.UNICODE,
)

# Fallback para súmulas do STJ que repetem o próprio número no fim do
# enunciado, tipo referência bibliográfica: "(SÚMULA 83, CORTE ESPECIAL, ...)".
SUMULA_NO_TEXTO = re.compile(r"S[ÚU]MULA\s+(\d+)", re.UNICODE)

# A base de desenvolvimento original não continha temas, mas bases canônicas
# posteriores podem registrá-los com ``natureza='tema'``. O número costuma
# aparecer no título como "Tema 339", "Tema repetitivo 1.076" ou "Tema de
# repercussão geral nº 123".
TEMA_NO_TEXTO = re.compile(
    r"\bTEMA(?:\s+REPETITIVO|\s+DE\s+REPERCUSS[ÃA]O\s+GERAL)?\s*"
    r"(?:N(?:\s*[.º°]|[UÚ]MERO)?\s*)?(\d+(?:\.\d+)*)",
    re.I | re.UNICODE,
)
TRIBUNAL_NO_TEXTO = re.compile(r"\b(STF|STJ|TST|TSE|STM)\b", re.I)

# Fallback legado para registros cujo título não esteja disponível ou legível.
SUMULA_SEM_NUMERO_NO_TEXTO = {
    int(chave): tuple(valor)
    for chave, valor in carregar_config("diplomas")["sumula_sem_numero_no_texto"].items()
}


def construir_indice_normativos(con: sqlite3.Connection) -> tuple[dict, list[dict]]:
    """Índice de leis, súmulas e temas por sua identidade jurídica.

    As chaves são ``lei|<estatuto>|<artigo>``,
    ``sumula|<tribunal>|<número>`` e ``tema|<tribunal>|<número>``.

    Aqui a chave é semântica, não posicional: não existe cabeçalho nesses
    registros, o alvo é o próprio enunciado. Preserva colisões para que uma
    chave com mais de um ID seja classificada como incompleta.
    """
    indice: dict[str, list[dict]] = {}
    sem_resolucao = []
    cur = con.execute(
        "SELECT documento_id, id, natureza, tribunal, texto FROM documentos "
        "WHERE natureza IN ('sumula','dispositivo','tema')"
    )
    for documento_id, id_canonico, natureza, tribunal, texto in cur:
        if natureza == "dispositivo":
            m = ARTIGO_INICIAL.match(texto)
            estatuto = ESTATUTO_POR_ID.get(id_canonico)
            if not m or not estatuto:
                sem_resolucao.append({"documento_id": documento_id, "id": id_canonico, "natureza": natureza})
                continue
            artigo = m.group(1).replace(".", "")
            chave = f"lei|{estatuto}|{artigo}"
        elif natureza == "sumula":
            primeira_linha = next((linha.strip() for linha in texto.splitlines() if linha.strip()), "")
            titulo = SUMULA_NO_TITULO.match(unicodedata.normalize("NFC", primeira_linha))
            m = SUMULA_NO_TEXTO.search(texto)
            if titulo:
                numero = normalizar_numero(titulo.group("numero"))
                if titulo.group("vinculante"):
                    tribunal = "STF"
                    numero = f"VINCULANTE-{numero}"
            elif m:
                numero = m.group(1)
            elif id_canonico in SUMULA_SEM_NUMERO_NO_TEXTO:
                tribunal, numero = SUMULA_SEM_NUMERO_NO_TEXTO[id_canonico]
            else:
                sem_resolucao.append({"documento_id": documento_id, "id": id_canonico, "natureza": natureza})
                continue
            chave = f"sumula|{tribunal}|{numero}"
        else:  # tema
            m = TEMA_NO_TEXTO.search(texto)
            tribunal_texto = TRIBUNAL_NO_TEXTO.search(texto)
            tribunal = tribunal or (tribunal_texto.group(1).upper() if tribunal_texto else None)
            if not m or not tribunal:
                sem_resolucao.append({"documento_id": documento_id, "id": id_canonico, "natureza": natureza})
                continue
            numero = normalizar_numero(m.group(1))
            chave = f"tema|{str(tribunal).upper()}|{numero}"
        indice.setdefault(chave, []).append({"id": id_canonico, "documento_id": documento_id})
    return indice, sem_resolucao


# --------------------------------------------------------------------------- #
# Resolução de citações contra o índice
# --------------------------------------------------------------------------- #

# construir_indice_acordaos usa a chave completa pra não colidir na base, mas
# uma citação não chega com "origem" pronta — só número e, às vezes, um
# pedaço da classe. As duas funções abaixo agrupam por número e desempatam
# com o que sobra. Deixei separadas de propósito, não como v1/v2, porque têm
# confiança diferente: resolver só usa o que a citação garante; a outra
# aposta quando nem isso resolve.

_NAO_ALFANUM = re.compile(r"[^a-z0-9]")


def _normalizar_para_comparar(s: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return _NAO_ALFANUM.sub("", sem_acento.lower())


_RUIDO_CLASSE_TSE = re.compile(
    r"^(?:.*?\bAC[ÓO]RD[ÃA]O\s+"
    r"|(?:PUBLICADO\s+EM\s+SESS[ÃA]O\s+)?(?:TRIBUNAL\.?\s+)?SUPERIOR\s+ELEITORAL\s+"
    r"|«\s*)",
    re.I,
)


def _classe_para_desempate(classe: str) -> str:
    """Remove preâmbulo do TSE antes de comparar a classe por extenso."""
    return _normalizar_para_comparar(_RUIDO_CLASSE_TSE.sub("", classe).strip())


TRIBUNAL_CITADO = re.compile(r"\b(?:" + "|".join(map(re.escape, carregar_config("tribunais")["siglas"])) + r")\b", re.I)
UF_CITADA = re.compile(r"(?:/|\s-\s|\(\s*)([A-Z]{2})(?=\b|\s*\))", re.I)
CLASSES_CITADAS = tuple(
    (re.compile(item["padrao"], re.I), item["classe"])
    for item in carregar_config("classes_processuais")["classes_citadas"]
)


def _classe_citada(trecho: str) -> str | None:
    sem_acento = unicodedata.normalize("NFKD", trecho).encode("ascii", "ignore").decode()
    for padrao, classe in CLASSES_CITADAS:
        if padrao.search(sem_acento):
            return classe
    return None


# Pares onde uma classe é uma especialização textual de outra (ex. "apelação
# criminal" é uma "apelação"). A citação às vezes usa só o termo genérico
# ("APL") sem informar a especialização — isso não é sinal de incompatibilidade,
# só falta de detalhe. Direção única: citação genérica aceita registro
# específico; citação que já especifica continua exigindo bater exatamente.
CLASSES_GENERICAS: dict[str, str] = carregar_config("classes_processuais").get("classes_genericas", {})


def _classe_compativel(classe_citada: str, classe_registro: str) -> bool:
    return classe_citada == classe_registro or CLASSES_GENERICAS.get(classe_registro) == classe_citada


# Estágio recursal (AgInt, AgRg, EDcl, embargos de divergência) dentro da
# mesma classe-base: "AgRg no REsp X" e "EDcl no AgRg no REsp X" são decisões
# diferentes do mesmo processo. O balde de classe (CLASSES_CITADAS) ignora
# esses prefixos, então dois registros colidentes por número quase sempre
# caem no mesmo balde. Isso complementa _classe_citada() sem substituí-lo.
MODIFICADORES_RECURSAIS = tuple(
    (re.compile(item["padrao"], re.I), item["modificador"])
    for item in carregar_config("classes_processuais")["modificadores_recursais"]
)


def _modificadores_citados(trecho: str) -> frozenset[str]:
    sem_acento = unicodedata.normalize("NFKD", trecho).encode("ascii", "ignore").decode()
    return frozenset(modificador for padrao, modificador in MODIFICADORES_RECURSAIS if padrao.search(sem_acento))


def _compativel(trecho: str, tribunal: str, classe: str, uf: str | None) -> bool:
    tribunal_citado = TRIBUNAL_CITADO.search(trecho)
    if tribunal_citado and tribunal_citado.group().upper() != tribunal:
        return False
    uf_citada = UF_CITADA.search(trecho)
    if uf_citada and uf and uf_citada.group(1).upper() != uf:
        return False
    classe_citada = _classe_citada(trecho)
    classe_registro = _classe_citada(classe)
    if classe_citada and classe_registro and not _classe_compativel(classe_citada, classe_registro):
        return False
    return True


def filtrar_candidatos(trecho: str, candidatos: set[int], indice_acordaos: dict,
                       con: sqlite3.Connection | None = None) -> set[int]:
    """Descarta registros incompatíveis com identificadores explícitos na citação."""
    metadados = {
        r["id"]: (tribunal, r["classe"], r["uf"])
        for (tribunal, _classe, _numero, _origem), registros in indice_acordaos.items()
        for r in registros if r["id"] in candidatos
    }
    if con and len(metadados) < len(candidatos):
        faltantes = candidatos - metadados.keys()
        for id_canonico, tribunal, texto in con.execute(
            f"SELECT id, tribunal, texto FROM documentos WHERE id IN ({','.join('?' for _ in faltantes)})",
            tuple(faltantes),
        ):
            campos = parsear_cabecalho(tribunal, texto)
            metadados[id_canonico] = (
                tribunal, campos["classe"] if campos else "",
                normalizar_uf(campos["uf"]) if campos else None,
            )
    return {
        id_canonico for id_canonico in candidatos
        if id_canonico not in metadados or _compativel(trecho, *metadados[id_canonico])
    }


def indice_por_numero(indice_acordaos: dict) -> dict[str, list[dict]]:
    por_numero: dict[str, list[dict]] = {}
    for (_tribunal, _classe, numero, _origem), registros in indice_acordaos.items():
        por_numero.setdefault(numero, []).extend(registros)
    return por_numero


def resolver(trecho: str, indice_acordaos: dict) -> set[int]:
    """Resolve uma citação contra o índice de acórdãos.

    Busca por número; se sobrar mais de um candidato com o mesmo número
    (mesmo processo, decisões diferentes), tento desempatar comparando a
    classe de cada candidato com o próprio texto da citação — às vezes ela
    já traz isso ("AgInt no Recurso Especial" bate só com um). Só decido se
    exatamente um bater; senão devolvo os candidatos como estão, sem
    forçar escolha.

    Devolve 1 id quando decide, mais de 1 quando a colisão sobrevive ao
    desempate, vazio quando o número não bate com nada do índice.
    """
    por_numero = indice_por_numero(indice_acordaos)
    tribunal_por_id = {
        r["id"]: tribunal
        for (tribunal, _classe, _numero, _origem), registros in indice_acordaos.items()
        for r in registros
    }
    for forma in u.variantes_numero(trecho):
        chave = normalizar_numero(forma) if u.CNJ.search(forma) else re.sub(r"[^0-9]", "", forma)
        registros = [r for r in por_numero.get(chave, []) if _compativel(
            trecho, tribunal_por_id[r["id"]], r["classe"], r["uf"])]
        if not registros:
            continue
        candidatos = {r["id"] for r in registros}
        if len(candidatos) == 1:
            return candidatos

        trecho_norm = _normalizar_para_comparar(trecho)
        batidas = {
            r["id"] for r in registros
            if _normalizar_para_comparar(r["classe"]) and _normalizar_para_comparar(r["classe"]) in trecho_norm
        }
        if len(batidas) == 1:
            return batidas
        return candidatos
    return set()


def resolver_com_aposta_de_tamanho(trecho: str, indice_acordaos: dict) -> set[int]:
    """Como resolver(), mas se ainda sobrar mais de um candidato aposto no
    de texto mais longo — a ideia é que entre um rascunho e a versão final
    republicada, a mais completa costuma prevalecer no acervo.

    Só confirmei essa aposta em 2 casos do dev set (2 de 2). Pouco pra
    confiar de olhos fechados — ainda não uso isso em produção sem revisar
    de novo com mais dados. Ao contrário de resolver(), esta função nunca
    devolve mais de 1 id: sempre aposta.
    """
    candidatos = resolver(trecho, indice_acordaos)
    if len(candidatos) <= 1:
        return candidatos

    por_numero = indice_por_numero(indice_acordaos)
    for forma in u.variantes_numero(trecho):
        chave = normalizar_numero(forma) if u.CNJ.search(forma) else re.sub(r"[^0-9]", "", forma)
        registros = por_numero.get(chave)
        if registros:
            mais_longo = max(registros, key=lambda r: r["texto_len"] or 0)
            return {mais_longo["id"]}
    return candidatos


# --------------------------------------------------------------------------- #
# Exportação
# --------------------------------------------------------------------------- #


def exportar_json(indice_acordaos: dict, indice_normativos: dict, caminho: str) -> None:
    """Grava o dicionário em JSON. Chave de acórdão vira string "TRIB|classe|
    número" (JSON não aceita tupla como chave); colisão fica marcada, não
    escondida atrás de "o último que venceu"."""
    acordaos_json = {}
    colisoes = []
    for (tribunal, classe, numero, origem), registros in indice_acordaos.items():
        chave = f"{tribunal}|{classe}|{numero}" + (f"|{origem}" if origem else "")
        if len(registros) > 1:
            colisoes.append({"chave": chave, "ids": [r["id"] for r in registros]})
        acordaos_json[chave] = registros if len(registros) > 1 else registros[0]

    saida = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acordaos": acordaos_json,
        "normativos": indice_normativos,
        "colisoes": colisoes,
    }
    with open(caminho, "w", encoding="utf-8") as fh:
        json.dump(saida, fh, ensure_ascii=False, indent=2, sort_keys=True)
