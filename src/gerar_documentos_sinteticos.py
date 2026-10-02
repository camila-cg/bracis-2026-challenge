"""Gera, por regras determinísticas, um corpus sintético auditável.

Cada documento derivado permanece no mesmo split do documento-base para evitar
vazamento entre treino, validação e teste.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import indice_utils as ind


MODELO_GERADOR = "procedural-regras-v1"
COLUNAS_GABARITO = (
    "nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho",
    "tipo", "classificacao", "id_canonico",
)


@dataclass(frozen=True)
class Anotacao:
    nivel: str
    documento_id: str
    citacao_id: str
    inicio: int
    fim: int
    trecho: str
    tipo: str
    classificacao: str
    id_canonico: str = ""


@dataclass(frozen=True)
class Referencia:
    trecho: str
    tipo: str
    classificacao: str
    id_canonico: str
    chave: str


def ler_gabarito(caminho: Path) -> dict[str, list[Anotacao]]:
    por_documento: dict[str, list[Anotacao]] = defaultdict(list)
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            anotacao = Anotacao(
                nivel=linha["nivel"], documento_id=linha["documento_id"],
                citacao_id=linha["citacao_id"], inicio=int(linha["inicio"]),
                fim=int(linha["fim"]), trecho=linha["trecho"], tipo=linha["tipo"],
                classificacao=linha["classificacao"], id_canonico=linha["id_canonico"],
            )
            por_documento[anotacao.documento_id].append(anotacao)
    for anotacoes in por_documento.values():
        anotacoes.sort(key=lambda item: (item.inicio, item.fim))
    return dict(por_documento)


def atribuir_splits(documentos: list[str], semente: int) -> dict[str, str]:
    """Divide por documento-base: aproximadamente 70%/15%/15%."""
    ids = sorted(documentos)
    random.Random(semente).shuffle(ids)
    quantidade = len(ids)
    treino = round(quantidade * 0.70)
    validacao = round(quantidade * 0.15)
    return {
        documento_id: (
            "train" if indice < treino else
            "validation" if indice < treino + validacao else "test"
        )
        for indice, documento_id in enumerate(ids)
    }


def _formatar_numero(numero: str) -> str:
    if "-" in numero and len(re.sub(r"\D", "", numero)) >= 14:
        sequencial, resto = numero.split("-", 1)
        digitos = re.sub(r"\D", "", resto)
        if len(digitos) == 13:
            return f"{int(sequencial)}-{digitos[:2]}.{digitos[2:6]}.{digitos[6]}.{digitos[7:9]}.{digitos[9:]}"
    digitos = re.sub(r"\D", "", numero)
    grupos = []
    while digitos:
        grupos.append(digitos[-3:])
        digitos = digitos[:-3]
    return ".".join(reversed(grupos))


def carregar_catalogo(base: Path) -> tuple[list[Referencia], set[str], set[int]]:
    con = sqlite3.connect(base.resolve().as_uri() + "?immutable=1", uri=True)
    try:
        acordaos, _ = ind.construir_indice_acordaos(con)
        normativos, _ = ind.construir_indice_normativos(con)
        referencias: list[Referencia] = []
        numeros_existentes: set[str] = set()
        ids_existentes = {linha[0] for linha in con.execute("SELECT id FROM documentos")}

        for (tribunal, classe, numero, _origem), registros in acordaos.items():
            numeros_existentes.add(re.sub(r"\D", "", numero))
            if len(registros) != 1:
                continue
            registro = registros[0]
            numero_exibido = _formatar_numero(numero)
            uf = f"/{registro['uf']}" if registro.get("uf") else ""
            trecho = f"{classe} nº {numero_exibido}{uf}, {tribunal}"
            referencias.append(Referencia(
                trecho=trecho, tipo="jurisprudencia", classificacao="real",
                id_canonico=str(registro["id"]),
                chave=f"acordao|{tribunal}|{numero}|{registro.get('uf') or ''}",
            ))

        for chave, registros in normativos.items():
            if len(registros) != 1:
                continue
            partes = chave.split("|", 2)
            if partes[0] == "sumula":
                trecho = f"Súmula {partes[2]} do {partes[1]}"
                tipo = "jurisprudencia"
            else:
                trecho = f"art. {partes[2]} do {partes[1]}"
                tipo = "lei"
            referencias.append(Referencia(
                trecho=trecho, tipo=tipo, classificacao="real",
                id_canonico=str(registros[0]["id"]), chave=chave,
            ))
        return referencias, numeros_existentes, ids_existentes
    finally:
        con.close()


def _trocar_primeiro_digito(trecho: str) -> tuple[str, str]:
    conversoes = {"0": "O", "1": "l", "5": "S"}
    for indice, caractere in enumerate(trecho):
        if caractere in conversoes:
            return trecho[:indice] + conversoes[caractere] + trecho[indice + 1:], "ocr_reversivel"
    return trecho, "ocr_sem_digito_elegivel"


def transformar_superficie(trecho: str, modo: int) -> tuple[str, str]:
    """Altera somente a superfície; não muda o identificador semântico."""
    if modo == 0:
        return trecho, "original"
    if modo == 1:
        novo = re.sub(r"\b(?:n[º°o]|n\.)\s*", "n. ", trecho, count=1, flags=re.I)
        return novo, "marcador_numero" if novo != trecho else "original"
    if modo == 2:
        novo = re.sub(r"(?<=[A-Za-zÀ-ÿ])\s+(?=\d)", "\n", trecho, count=1)
        return novo, "quebra_linha" if novo != trecho else "original"
    if modo == 3:
        novo = re.sub(r"/([A-Z]{2})\b", r" - \1", trecho, count=1)
        return novo, "uf_hifen" if novo != trecho else "original"
    if modo == 4:
        return _trocar_primeiro_digito(trecho)
    if modo == 5:
        return unicodedata.normalize("NFD", trecho), "unicode_nfd"
    if modo == 6:
        novo = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", trecho, count=1)
        return novo, "numero_sem_separador" if novo != trecho else "original"
    novo = re.sub(r"\s+", "  ", trecho, count=1)
    return novo, "espacamento_duplo" if novo != trecho else "original"


def _mutar_para_inventada(
    referencia: Referencia, numeros_existentes: set[str], rng: random.Random,
) -> Referencia:
    candidatos = list(re.finditer(r"\d", referencia.trecho))
    if not candidatos:
        return Referencia(
            trecho="Recurso Especial nº 9.999.999/ZZ", tipo="jurisprudencia",
            classificacao="inventada", id_canonico="", chave="inventada|fallback",
        )
    ordem = list(range(len(candidatos)))
    rng.shuffle(ordem)
    for indice_match in ordem:
        match = candidatos[indice_match]
        atual = int(match.group())
        for incremento in range(1, 10):
            novo_digito = str((atual + incremento) % 10)
            mutado = referencia.trecho[:match.start()] + novo_digito + referencia.trecho[match.end():]
            blocos = [re.sub(r"\D", "", bloco) for bloco in re.findall(r"\d[\d.\-/]*", mutado)]
            numero_principal = max(blocos, key=len, default="")
            if numero_principal and numero_principal not in numeros_existentes:
                return Referencia(
                    trecho=mutado, tipo=referencia.tipo, classificacao="inventada",
                    id_canonico="", chave=f"inventada|{referencia.chave}|digito-{match.start()}",
                )
    raise RuntimeError(f"não foi possível criar referência inventada a partir de {referencia.trecho!r}")


def _referencia_incompleta(tipo: str, rng: random.Random) -> Referencia:
    if tipo == "lei":
        opcoes = ["artigo aplicável da legislação federal", "dispositivo pertinente do CPC"]
    else:
        opcoes = [
            "precedente do STJ sobre a matéria",
            "julgado do STF proferido recentemente",
            "entendimento consolidado do tribunal superior",
        ]
    trecho = rng.choice(opcoes)
    return Referencia(trecho, tipo, "incompleta", "", f"incompleta|{trecho}")


def _referencia_inventada_lei() -> Referencia:
    return Referencia(
        trecho="art. 99999 do Código Civil", tipo="lei",
        classificacao="inventada", id_canonico="",
        chave="inventada|lei|Código Civil|99999",
    )


def _renderizar_base(
    texto: str, anotacoes: list[Anotacao], variante: int, remover_citacoes: bool,
) -> tuple[str, list[dict], list[str]]:
    partes: list[str] = []
    novas: list[dict] = []
    transformacoes: list[str] = []
    cursor = 0
    for indice, anotacao in enumerate(anotacoes):
        if anotacao.inicio < cursor:
            raise ValueError(f"anotações sobrepostas em {anotacao.documento_id}")
        original = texto[anotacao.inicio:anotacao.fim]
        partes.append(texto[cursor:anotacao.inicio])
        if remover_citacoes:
            substituto = "referência jurídica pertinente"
            partes.append(substituto)
            cursor = anotacao.fim
            continue
        substituto, transformacao = transformar_superficie(original, (variante + indice) % 8)
        inicio = sum(map(len, partes))
        partes.append(substituto)
        fim = inicio + len(substituto)
        novas.append({
            "nivel": anotacao.nivel, "citacao_id_origem": anotacao.citacao_id,
            "inicio": inicio, "fim": fim, "trecho": substituto,
            "tipo": anotacao.tipo, "classificacao": anotacao.classificacao,
            "id_canonico": anotacao.id_canonico,
            "transformacao": transformacao,
        })
        transformacoes.append(transformacao)
        cursor = anotacao.fim
    partes.append(texto[cursor:])
    return "".join(partes), novas, transformacoes


def _adicionar_distratores(texto: str, rng: random.Random) -> tuple[str, list[str]]:
    valor = f"{rng.randint(1_000, 900_000):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    prefixo = (
        f"Protocolo interno: {rng.randint(100000, 999999)}\n"
        f"Valor da causa: {valor}"
        f"\nOAB: {rng.randint(10_000, 99_999)}/SP · fls. {rng.randint(2, 800)}/{rng.randint(801, 999)}\n\n"
    )
    return prefixo + texto, ["protocolo", "valor_da_causa_sem_cifrao", "oab", "folhas"]


def _acrescentar_referencia(
    texto: str, anotacoes: list[dict], referencia: Referencia, nivel: str,
) -> tuple[str, list[dict]]:
    introducao = (
        "\n\nComo fundamento adicional, invoca-se " if referencia.tipo == "jurisprudencia"
        else "\n\nTambém incide "
    )
    final = ", por tratar diretamente da matéria examinada.\n"
    inicio = len(texto) + len(introducao)
    texto = texto + introducao + referencia.trecho + final
    anotacoes.append({
        "nivel": nivel, "citacao_id_origem": None, "inicio": inicio,
        "fim": inicio + len(referencia.trecho), "trecho": referencia.trecho,
        "tipo": referencia.tipo, "classificacao": referencia.classificacao,
        "id_canonico": referencia.id_canonico, "chave_catalogo": referencia.chave,
        "transformacao": "insercao_catalogo",
    })
    return texto, anotacoes


def gerar_documento(
    texto_base: str, anotacoes_base: list[Anotacao], variante: int,
    referencias: list[Referencia], numeros_existentes: set[str], semente: int,
) -> tuple[str, list[dict], dict]:
    rng = random.Random(semente)
    perfil = ("superficie", "real_extra", "inventada_extra", "incompleta_extra",
              "distratores", "sem_citacoes")[variante % 6]
    remover = perfil == "sem_citacoes"
    texto, anotacoes, transformacoes = _renderizar_base(
        texto_base, anotacoes_base, variante, remover,
    )
    distratores: list[str] = []
    if perfil == "distratores":
        tamanho_antes = len(texto)
        texto, distratores = _adicionar_distratores(texto, rng)
        deslocamento = len(texto) - tamanho_antes
        for anotacao in anotacoes:
            anotacao["inicio"] += deslocamento
            anotacao["fim"] += deslocamento

    if perfil in {"real_extra", "inventada_extra", "incompleta_extra"}:
        tipo_desejado = "lei" if (semente % 3 == 0) else "jurisprudencia"
        referencias_tipo = [item for item in referencias if item.tipo == tipo_desejado]
        referencia_real = rng.choice(referencias_tipo or referencias)
        if perfil == "inventada_extra":
            if tipo_desejado == "lei":
                referencia = _referencia_inventada_lei()
            else:
                acordaos = [item for item in referencias_tipo if item.chave.startswith("acordao|")]
                referencia = _mutar_para_inventada(rng.choice(acordaos), numeros_existentes, rng)
        elif perfil == "incompleta_extra":
            referencia = _referencia_incompleta(referencia_real.tipo, rng)
        else:
            referencia = referencia_real
        texto, anotacoes = _acrescentar_referencia(
            texto, anotacoes, referencia, anotacoes_base[0].nivel,
        )
    return texto, anotacoes, {
        "perfil": perfil, "transformacoes": Counter(transformacoes),
        "distratores": distratores,
    }


def validar_documento(
    documento_id: str, texto: str, anotacoes: list[dict], ids_existentes: set[int],
) -> list[str]:
    erros: list[str] = []
    fim_anterior = 0
    for anotacao in sorted(anotacoes, key=lambda item: item["inicio"]):
        inicio, fim = anotacao["inicio"], anotacao["fim"]
        if not (0 <= inicio < fim <= len(texto)):
            erros.append(f"{documento_id}: span fora do texto: {inicio}:{fim}")
            continue
        if inicio < fim_anterior:
            erros.append(f"{documento_id}: spans sobrepostos em {inicio}")
        if texto[inicio:fim] != anotacao["trecho"]:
            erros.append(f"{documento_id}: trecho diverge do span em {inicio}:{fim}")
        if anotacao["classificacao"] == "real":
            if not anotacao["id_canonico"] or int(anotacao["id_canonico"]) not in ids_existentes:
                erros.append(f"{documento_id}: citação real sem id canônico válido")
        elif anotacao["id_canonico"]:
            erros.append(f"{documento_id}: citação não real com id canônico")
        fim_anterior = max(fim_anterior, fim)
    return erros


def executar(
    pasta_txt: Path, goldenset: Path, base: Path, saida: Path,
    variacoes: int, semente: int,
) -> dict:
    if variacoes < 1:
        raise ValueError("--variacoes deve ser pelo menos 1")
    if saida.exists() and any(saida.iterdir()):
        raise FileExistsError(f"a pasta de saída não está vazia: {saida}")

    por_documento = ler_gabarito(goldenset)
    arquivos = {arquivo.stem: arquivo for arquivo in pasta_txt.glob("*.txt")}
    faltantes = sorted(set(por_documento) - arquivos.keys())
    if faltantes:
        raise FileNotFoundError(f"documentos sem .txt: {faltantes}")
    splits = atribuir_splits(list(por_documento), semente)
    referencias, numeros_existentes, ids_existentes = carregar_catalogo(base)

    (saida / "txt").mkdir(parents=True, exist_ok=True)
    linhas_gabarito: list[dict] = []
    manifestos: list[dict] = []
    erros: list[str] = []
    contagem = Counter()

    for documento_origem in sorted(por_documento):
        texto_base = arquivos[documento_origem].read_text(encoding="utf-8")
        anotacoes_base = por_documento[documento_origem]
        split = splits[documento_origem]
        pasta_split = saida / "txt" / split
        pasta_split.mkdir(parents=True, exist_ok=True)
        for variante in range(variacoes):
            semente_documento = int(hashlib.sha256(
                f"{semente}|{documento_origem}|{variante}".encode()
            ).hexdigest()[:16], 16)
            documento_id = f"syn_{split}_{documento_origem}_v{variante + 1:03d}"
            texto, anotacoes, metadados = gerar_documento(
                texto_base, anotacoes_base, variante, referencias,
                numeros_existentes, semente_documento,
            )
            erros.extend(validar_documento(documento_id, texto, anotacoes, ids_existentes))
            (pasta_split / f"{documento_id}.txt").write_text(texto, encoding="utf-8", newline="\n")
            for indice, anotacao in enumerate(anotacoes, 1):
                linha = {
                    "nivel": anotacao["nivel"], "documento_id": documento_id,
                    "citacao_id": f"{documento_id}_c{indice:03d}",
                    "inicio": anotacao["inicio"], "fim": anotacao["fim"],
                    "trecho": anotacao["trecho"], "tipo": anotacao["tipo"],
                    "classificacao": anotacao["classificacao"],
                    "id_canonico": anotacao["id_canonico"],
                }
                linhas_gabarito.append(linha)
                contagem[(split, anotacao["tipo"], anotacao["classificacao"])] += 1
            manifestos.append({
                "documento_id": documento_id, "documento_origem": documento_origem,
                "split": split, "variante": variante + 1,
                "semente": semente_documento, "modelo": MODELO_GERADOR,
                "perfil": metadados["perfil"],
                "transformacoes": dict(metadados["transformacoes"]),
                "distratores": metadados["distratores"],
                "quantidade_citacoes": len(anotacoes),
                "citacoes": [
                    {
                        "citacao_id_origem": item.get("citacao_id_origem"),
                        "inicio": item["inicio"], "fim": item["fim"],
                        "tipo": item["tipo"], "classificacao": item["classificacao"],
                        "id_canonico": item["id_canonico"],
                        "transformacao": item.get("transformacao"),
                        "chave_catalogo": item.get("chave_catalogo"),
                    }
                    for item in anotacoes
                ],
            })

    if erros:
        raise ValueError("falha na validação:\n" + "\n".join(erros[:20]))

    with (saida / "goldenset_offsets.csv").open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_GABARITO)
        escritor.writeheader()
        escritor.writerows(linhas_gabarito)
    with (saida / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as arquivo:
        for manifesto in manifestos:
            arquivo.write(json.dumps(manifesto, ensure_ascii=False, sort_keys=True) + "\n")

    relatorio = {
        "modelo": MODELO_GERADOR, "semente": semente,
        "variacoes_por_documento": variacoes,
        "documentos_origem": len(por_documento),
        "documentos_gerados": len(manifestos),
        "citacoes_geradas": len(linhas_gabarito),
        "splits_documentos_origem": dict(Counter(splits.values())),
        "splits_documentos_gerados": dict(Counter(item["split"] for item in manifestos)),
        "distribuicao_citacoes": {
            "|".join(chave): valor for chave, valor in sorted(contagem.items())
        },
        "catalogo_referencias_reais": len(referencias),
        "validacao": {"erros": 0, "status": "ok"},
    }
    (saida / "relatorio_validacao.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return relatorio


def main() -> None:
    raiz = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt", type=Path, default=raiz / "data" / "txt")
    parser.add_argument("--goldenset", type=Path, default=raiz / "data" / "goldenset_offsets.csv")
    parser.add_argument("--db", type=Path, default=raiz / "data" / "desafio1_bracis.db")
    parser.add_argument("--saida", type=Path, default=raiz / "data" / "sinteticos")
    parser.add_argument("--variacoes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    relatorio = executar(
        args.txt, args.goldenset, args.db, args.saida, args.variacoes, args.seed,
    )
    print(json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
