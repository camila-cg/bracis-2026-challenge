"""ETAPA 2 — remove campos de metadados e distratores com base no contexto."""

from __future__ import annotations

import re
from dataclasses import dataclass

import indice_utils as ind
from steps.pipeline_utils import corrigir_ocr


@dataclass(frozen=True, slots=True)
class CampoMetadado:
    rotulo: str
    inicio: int
    fim: int
    numero: str | None = None


@dataclass(frozen=True, slots=True)
class ContextoDocumento:
    campos: tuple[CampoMetadado, ...]
    processo_principal: str | None
    corpo_inicio: int


ROTULO = re.compile(
    r"^\s*(?P<rotulo>processo|autos|reclamad[ao]|reclamante|agravad[ao]|agravante"
    r"|recorrid[ao]|recorrente|relator[a]?|apelad[aoã]|apelante|assistido"
    r"|representant[ec]|representad[oc]|rcpresentant[ec]|rcpresentad[oc]"
    r"|impetrante|paciente|autoridade\s+coatora|interessad[ao]|assunto|refer[êe]ncia"
    r"|elaborado\s+por|memorial|valor\s+(?:atribu[ií]do\s+)?(?:da|à)\s+causa)"
    r"\b\s*(?:n\s*[º°o.]?\s*)?:?\s*",
    re.I,
)
TITULO_CORPO = re.compile(
    r"^\s*(?:(?:[IVXLCDM]+|\d{1,2})\s*[-–—.)]\s*)?"
    r"(?:AC[ÓO]RD[ÃA]O|DECIS[ÃA]O(?:\s+MONOCR[ÁA]TICA)?|MEMORIAL|EMENTA|RELAT[ÓO]RIO|VOTO|FUNDAMENTA[ÇC][ÃA]O)\s*$",
    re.I,
)
# Fallback estrutural para rótulos de cabeçalho fora da enumeração de ROTULO
# (ex. "Embargante:", "Exequente:", "Executado:", "Denunciado:") e para ruído
# de OCR dentro do próprio rótulo (ex. "Apelãdo:"). Só é consultado quando
# ROTULO não bate — nunca compete com os casos especiais de processo/autos/
# valor da causa, que continuam resolvidos só por ROTULO. Exige maiúscula
# inicial pra reduzir a chance de casar conectivo de prosa em minúscula.
ROTULO_GENERICO = re.compile(r"^\s*(?P<rotulo>[A-ZÀ-Ú][^\W\d_]*(?:[ ][^\W\d_]+){0,3})\s*:")
NUMERO_PROCESSO = re.compile(
    r"(?:TST-[A-Z-]+-)?\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}"
    r"|\d{5,}(?:[./-]\d+)*",
    re.I,
)
# Folhas dos autos: "fls. 234/567", mas também "fls 234", "fl. 12", "f. 45" e
# "folhas 23". "f" sozinho só com ponto, para não pegar letra solta antes de número.
FOLHAS = r"\b(?:fls?\.?|f\.|folhas?)\s*\d"
DISTRATOR = re.compile(rf"R\$|{FOLHAS}|\bOAB\b|\bprotocolo\b", re.I)
_RUBRICA_VALOR = (
    r"(?:valor\s+(?:atribu[ií]do\s+)?(?:da|à)\s+causa"
    r"|d[aá]-se\s+à\s+causa\s+o\s+valor)"
)
_QUANTIA = r"(?:R\$\s*)?(?P<quantia>\d{1,3}(?:\.\d{3})+(?:,\d{2})?|\d+(?:,\d{2})?)\b"
VALOR_DA_CAUSA = re.compile(rf"{_RUBRICA_VALOR}\s*(?::|de)?\s*{_QUANTIA}", re.I)
# "Valor da causa:" sozinho na linha, sem a quantia — ela vem na linha seguinte.
ROTULO_VALOR_SEM_QUANTIA = re.compile(rf"^\s*{_RUBRICA_VALOR}\s*(?::|de)?\s*$", re.I)
QUANTIA_NO_INICIO_DA_LINHA = re.compile(rf"^\s*{_QUANTIA}", re.I)


def analisar_contexto(texto: str) -> ContextoDocumento:
    """Lê somente o bloco inicial de metadados; offsets seguem o texto original."""
    campos: list[CampoMetadado] = []
    numeros_principais: set[str] = set()
    offset = 0
    corpo_inicio = len(texto)
    encontrou_campo = False
    for linha_com_quebra in texto.splitlines(keepends=True):
        linha = linha_com_quebra.rstrip("\r\n")
        if TITULO_CORPO.fullmatch(linha) and encontrou_campo:
            corpo_inicio = offset
            break
        m = ROTULO.match(linha)
        if m:
            rotulo_bruto = re.sub(r"\s+", " ", m.group("rotulo").lower())
            separador = linha[m.end("rotulo"):].lstrip()
            if rotulo_bruto not in {"processo", "autos", "memorial"} and not rotulo_bruto.startswith("valor") and not separador.startswith(":"):
                m = None
        if not m and encontrou_campo:
            # Só consultado depois que pelo menos um campo já foi reconhecido
            # por ROTULO — ou seja, já sabemos que estamos dentro de um bloco
            # de cabeçalho de verdade. Sem essa trava, a primeira linha de um
            # texto qualquer com "Palavra: citação" (ex. "Antes: REsp .../PR")
            # seria tratada como cabeçalho por engano.
            m = ROTULO_GENERICO.match(linha)
            if m:
                rotulo_bruto = re.sub(r"\s+", " ", m.group("rotulo").lower())
        if m:
            encontrou_campo = True
            rotulo = rotulo_bruto
            numero = None
            fim = offset + len(linha)
            if rotulo in {"processo", "autos"}:
                achado = NUMERO_PROCESSO.search(linha, m.end())
                if achado:
                    numero = ind.normalizar_numero(achado.group())
                    numeros_principais.add(numero)
                    # Uma citação posterior na mesma linha não integra o campo.
                    fim = offset + achado.end()
                else:
                    fim = offset + m.end()
            elif rotulo.startswith("valor"):
                valor = VALOR_DA_CAUSA.search(corrigir_ocr(linha))
                if valor:
                    fim = offset + valor.end()
            campos.append(CampoMetadado(rotulo, offset + m.start(), fim, numero))
        elif linha.strip() and encontrou_campo:
            # Uma frase inicia o corpo mesmo sem título explícito.
            if linha != linha.upper():
                corpo_inicio = offset
                break
        elif linha.strip() and not encontrou_campo and linha != linha.upper():
            corpo_inicio = offset
            break
        offset += len(linha_com_quebra)
    principal = next(iter(numeros_principais)) if len(numeros_principais) == 1 else None
    return ContextoDocumento(tuple(campos), principal, corpo_inicio)


def _e_valor_da_causa(candidato: dict, texto: str) -> bool:
    inicio, fim = candidato["inicio"], candidato["fim"]
    if not (0 <= inicio < fim <= len(texto)):
        return False
    linha_inicio = texto.rfind("\n", 0, inicio) + 1
    linha_fim = texto.find("\n", fim)
    if linha_fim < 0:
        linha_fim = len(texto)
    # corrigir_ocr troca 1 caractere por 1 caractere (letra->dígito colada a
    # dígito): não desloca offsets, então dá pra reaproveitar as posições da
    # linha original mesmo depois da correção.
    linha = corrigir_ocr(texto[linha_inicio:linha_fim])
    for campo in VALOR_DA_CAUSA.finditer(linha):
        quantia_inicio = linha_inicio + campo.start("quantia")
        campo_fim = linha_inicio + campo.end()
        if inicio < campo_fim and fim > quantia_inicio and fim <= campo_fim:
            return True
    # Rótulo e quantia às vezes ficam em linhas diferentes
    # ("Valor da causa:\nR$ 10.000,00"): sem isso, a quantia sobra como se
    # fosse uma citação. Só olha 1 linha pra trás, e só quando ela é
    # exclusivamente o rótulo (nada de conteúdo estranho vira gatilho).
    if linha_inicio > 0:
        anterior_fim = linha_inicio - 1
        anterior_inicio = texto.rfind("\n", 0, anterior_fim) + 1
        linha_anterior = corrigir_ocr(texto[anterior_inicio:anterior_fim])
        if ROTULO_VALOR_SEM_QUANTIA.search(linha_anterior):
            m = QUANTIA_NO_INICIO_DA_LINHA.match(linha)
            if m:
                quantia_inicio = linha_inicio + m.start("quantia")
                campo_fim = linha_inicio + m.end()
                if inicio < campo_fim and fim > quantia_inicio and fim <= campo_fim:
                    return True
    return False


def motivo_distrator(candidato: dict, texto: str | None = None,
                     contexto: ContextoDocumento | None = None) -> str | None:
    """Explica a exclusão; ausência de evidência preserva o candidato."""
    if DISTRATOR.search(candidato["trecho"]):
        return "distrator_lexical"
    if texto is None:
        return None
    if _e_valor_da_causa(candidato, texto):
        return "valor_da_causa"
    contexto = contexto or analisar_contexto(texto)
    inicio = candidato["inicio"]
    fim = candidato.get("fim", inicio + len(candidato["trecho"]))
    if inicio >= contexto.corpo_inicio:
        return None
    for campo in contexto.campos:
        if not (campo.inicio <= inicio < campo.fim):
            continue
        # O extrator pode reunir dois campos adjacentes em um único span.
        # Ele continua sendo metadado quando termina dentro do último campo.
        if fim > campo.fim and not any(
            outro.fim >= fim for outro in contexto.campos if outro.inicio > campo.inicio
        ):
            continue
        if campo.rotulo in {"processo", "autos"}:
            if campo.numero and campo.numero == contexto.processo_principal:
                return "campo_processo_principal"
            if campo.numero and contexto.processo_principal is None:
                return "campo_processo"
            continue
        return "campo_" + campo.rotulo.replace(" ", "_")
    return None


def e_distrator(candidato: dict, texto: str | None = None,
                contexto: ContextoDocumento | None = None) -> bool:
    return motivo_distrator(candidato, texto, contexto) is not None


def filtrar_distratores(candidatos: list[dict], texto: str | None = None) -> list[dict]:
    contexto = analisar_contexto(texto) if texto is not None else None
    return [c for c in candidatos if not e_distrator(c, texto, contexto)]
