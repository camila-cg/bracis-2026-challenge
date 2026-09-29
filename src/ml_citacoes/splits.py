"""Preparação reproduzível de dados reais e sintéticos, sem vazamento nos folds."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path


def validar_origem_original(*caminhos: Path) -> None:
    proibidos = ("sintetico", "sintético", "synthetic")
    for caminho in caminhos:
        texto = str(caminho).lower()
        if any(termo in texto for termo in proibidos):
            raise ValueError(f"dataset sintético não é permitido: {caminho}")


def ler_dataset_original(pasta_txt: Path, goldenset: Path) -> list[dict]:
    validar_origem_original(pasta_txt, goldenset)
    por_documento: dict[str, list[dict]] = {}
    with goldenset.open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        obrigatorias = {"documento_id", "inicio", "fim", "trecho", "tipo"}
        if not obrigatorias.issubset(leitor.fieldnames or []):
            raise ValueError(f"goldenset sem colunas obrigatórias: {sorted(obrigatorias)}")
        for linha in leitor:
            documento_id = linha["documento_id"]
            if documento_id.startswith("syn_"):
                raise ValueError(f"documento sintético não é permitido: {documento_id}")
            por_documento.setdefault(documento_id, []).append({
                "inicio": int(linha["inicio"]),
                "fim": int(linha["fim"]),
                "trecho": linha["trecho"].replace("\\n", "\n"),
                "tipo": linha["tipo"],
                "classificacao": linha.get("classificacao", ""),
                "id_canonico": linha.get("id_canonico", ""),
                "nivel": linha.get("nivel", ""),
            })
    documentos: list[dict] = []
    for documento_id, spans in sorted(por_documento.items()):
        caminho = pasta_txt / f"{documento_id}.txt"
        bruto = caminho.read_bytes()
        if b"\r\n" in bruto:
            raise ValueError(f"{caminho}: CRLF deslocaria os offsets")
        texto = bruto.decode("utf-8")
        spans.sort(key=lambda item: (item["inicio"], item["fim"]))
        for span in spans:
            if texto[span["inicio"]:span["fim"]] != span["trecho"]:
                raise ValueError(f"{documento_id}: trecho diverge em {span['inicio']}:{span['fim']}")
        nivel = str(spans[0].get("nivel") or ("2" if "_n2_" in documento_id else "1"))
        documentos.append({
            "documento_id": documento_id,
            "texto": texto,
            "spans": spans,
            "nivel": nivel,
            "origem": "original",
        })
    return documentos


def ler_dataset_sintetico(raiz: Path) -> list[dict]:
    """Lê um pacote sintético validado, incluindo documentos sem citações."""
    raiz = raiz.resolve()
    manifesto_path = raiz / "manifest.csv"
    goldenset_path = raiz / "goldenset_offsets.csv"
    if not manifesto_path.is_file() or not goldenset_path.is_file():
        raise ValueError(f"pacote sintético incompleto: {raiz}")

    metadados: dict[str, dict[str, str]] = {}
    with manifesto_path.open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        obrigatorias = {"documento_id", "arquivo", "nivel"}
        if not obrigatorias.issubset(leitor.fieldnames or []):
            raise ValueError(f"manifesto sintético sem colunas obrigatórias: {sorted(obrigatorias)}")
        for linha in leitor:
            documento_id = linha["documento_id"]
            if documento_id in metadados:
                raise ValueError(f"documento sintético duplicado: {documento_id}")
            metadados[documento_id] = linha

    por_documento: dict[str, list[dict]] = {documento_id: [] for documento_id in metadados}
    with goldenset_path.open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        obrigatorias = {"documento_id", "inicio", "fim", "trecho", "tipo", "classificacao"}
        if not obrigatorias.issubset(leitor.fieldnames or []):
            raise ValueError(f"goldenset sintético sem colunas obrigatórias: {sorted(obrigatorias)}")
        for linha in leitor:
            documento_id = linha["documento_id"]
            if documento_id not in metadados:
                raise ValueError(f"anotação sintética fora do manifesto: {documento_id}")
            por_documento[documento_id].append({
                "inicio": int(linha["inicio"]),
                "fim": int(linha["fim"]),
                "trecho": linha["trecho"].replace("\\n", "\n"),
                "tipo": linha["tipo"],
                "classificacao": linha.get("classificacao", ""),
                "id_canonico": linha.get("id_canonico", ""),
                "nivel": linha.get("nivel", ""),
            })

    documentos = []
    for documento_id, meta in sorted(metadados.items()):
        caminho = (raiz / meta["arquivo"]).resolve()
        if raiz not in caminho.parents or not caminho.is_file():
            raise ValueError(f"arquivo sintético inválido: {meta['arquivo']}")
        bruto = caminho.read_bytes()
        if b"\r\n" in bruto:
            raise ValueError(f"{caminho}: CRLF deslocaria os offsets")
        texto = bruto.decode("utf-8")
        spans = sorted(por_documento[documento_id], key=lambda item: (item["inicio"], item["fim"]))
        fim_anterior = 0
        for span in spans:
            if span["inicio"] < fim_anterior:
                raise ValueError(f"{documento_id}: spans sintéticos sobrepostos")
            if texto[span["inicio"]:span["fim"]] != span["trecho"]:
                raise ValueError(f"{documento_id}: trecho diverge em {span['inicio']}:{span['fim']}")
            fim_anterior = span["fim"]
        documentos.append({
            "documento_id": documento_id,
            "texto": texto,
            "spans": spans,
            "nivel": str(meta["nivel"]),
            "origem": "sintetico",
        })
    return documentos


def criar_folds(documentos: list[dict], n_folds: int = 5, seed: int = 260926) -> list[dict]:
    """Distribuição gulosa determinística por dificuldade, tipo, classe e volume."""
    if len(documentos) < n_folds:
        raise ValueError("há menos documentos que folds")
    rng = random.Random(seed)
    itens = list(documentos)
    rng.shuffle(itens)
    itens.sort(key=lambda doc: _peso(_caracteristicas(doc)), reverse=True)
    folds: list[list[dict]] = [[] for _ in range(n_folds)]
    totais = [Counter() for _ in range(n_folds)]
    for documento in itens:
        caracteristicas = _caracteristicas(documento)
        indice = min(
            range(n_folds),
            key=lambda i: (_custo(totais[i], caracteristicas), len(folds[i]), i),
        )
        folds[indice].append(documento)
        totais[indice].update(caracteristicas)
    todos_ids = {doc["documento_id"] for doc in documentos}
    resultado = []
    for indice, validacao in enumerate(folds):
        ids_validacao = sorted(doc["documento_id"] for doc in validacao)
        ids_treino = sorted(todos_ids - set(ids_validacao))
        validar_sem_vazamento(ids_treino, ids_validacao)
        resultado.append({
            "fold": indice,
            "seed": seed,
            "treino": ids_treino,
            "validacao": ids_validacao,
            "caracteristicas_validacao": dict(totais[indice]),
        })
    return resultado


def validar_sem_vazamento(treino: list[str], validacao: list[str]) -> None:
    repetidos = set(treino) & set(validacao)
    if repetidos:
        raise ValueError(f"vazamento entre treino e validação: {sorted(repetidos)}")


def preparar(
    pasta_txt: Path,
    goldenset: Path,
    saida: Path,
    *,
    n_folds: int = 5,
    seed: int = 260926,
    sinteticos: list[Path] | None = None,
) -> None:
    originais = ler_dataset_original(pasta_txt, goldenset)
    folds = criar_folds(originais, n_folds, seed)
    fontes_sinteticas = [Path(caminho).resolve() for caminho in (sinteticos or [])]
    documentos_sinteticos = [
        documento
        for fonte in fontes_sinteticas
        for documento in ler_dataset_sintetico(fonte)
    ]
    ids_originais = {doc["documento_id"] for doc in originais}
    ids_sinteticos = [doc["documento_id"] for doc in documentos_sinteticos]
    if len(ids_sinteticos) != len(set(ids_sinteticos)):
        raise ValueError("documento sintético repetido entre pacotes")
    colisao = ids_originais & set(ids_sinteticos)
    if colisao:
        raise ValueError(f"IDs repetidos entre dados originais e sintéticos: {sorted(colisao)}")

    # Sintéticos enriquecem apenas o treino. A validação de cada fold continua
    # composta exclusivamente por documentos originais da competição.
    for fold in folds:
        fold["treino"] = sorted([*fold["treino"], *ids_sinteticos])
        fold["sinteticos_treino"] = sorted(ids_sinteticos)

    documentos = [*originais, *documentos_sinteticos]
    saida.mkdir(parents=True, exist_ok=True)
    with (saida / "documentos.jsonl").open("w", encoding="utf-8", newline="\n") as arquivo:
        for documento in documentos:
            arquivo.write(json.dumps(documento, ensure_ascii=False) + "\n")
    manifesto = {
        "origem": "dataset_original_competicao+sinteticos" if documentos_sinteticos else "dataset_original_competicao",
        "aumento_aplicado": False,
        "seed": seed,
        "n_folds": n_folds,
        "documentos_originais": len(originais),
        "documentos_sinteticos": len(documentos_sinteticos),
        "fontes_sinteticas": [str(caminho) for caminho in fontes_sinteticas],
        "politica_sinteticos": "somente_treino;validacao_apenas_original",
        "folds": folds,
    }
    (saida / "folds.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")


def _caracteristicas(documento: dict) -> Counter:
    spans = documento["spans"]
    nivel = str(documento.get("nivel") or (spans[0].get("nivel") if spans else "") or ("2" if "_n2_" in documento["documento_id"] else "1"))
    contagem = Counter({"documentos": 1, "citacoes": len(spans), f"nivel_{nivel}": 1})
    if nivel == "2":
        contagem["ocr"] = 1
    for span in spans:
        contagem[f"tipo_{span['tipo']}"] += 1
        if span.get("classificacao"):
            contagem[f"classe_{span['classificacao']}"] += 1
    return contagem


def _peso(caracteristicas: Counter) -> int:
    return 5 * caracteristicas["citacoes"] + 3 * caracteristicas["ocr"] + len(caracteristicas)


def _custo(total: Counter, adicao: Counter) -> float:
    pesos = {"documentos": 20.0, "citacoes": 2.0, "ocr": 8.0}
    return sum((total[chave] + valor) * pesos.get(chave, 3.0) for chave, valor in adicao.items())
