"""Gera e audita um manifesto determinístico dos datasets do projeto.

O script não cria splits nem copia dados. Cada dataset precisa declarar sua
origem e seu papel explicitamente em uma especificação JSON, por exemplo::

    {
      "versao": 1,
      "datasets": [
        {
          "nome": "desenvolvimento",
          "documentos": "data/txt",
          "anotacoes": "data/goldenset_offsets.csv",
          "origem": "desafio-bracis",
          "papel": "treino"
        }
      ]
    }

Os caminhos da especificação devem ser relativos à raiz auditada. São aceitos
``treino``, ``validacao``/``validação`` e ``teste``; os aliases ingleses
``train``, ``validation`` e ``test`` também são normalizados para esses valores.

Uso a partir da raiz do repositório::

    python src/validacao/manifesto_datasets.py gerar \
        --especificacao caminho/especificacao.json \
        --saida caminho/manifesto.json

    python src/validacao/manifesto_datasets.py auditar \
        --manifesto caminho/manifesto.json

``auditar`` recalcula todo o conteúdo e falha se arquivos, hashes, tamanhos,
contagens ou metadados divergirem. Tanto ``gerar`` quanto ``auditar`` falham se
o mesmo ID de documento ou o mesmo conteúdo (SHA-256 exato) aparecer em papéis
distintos. Sobreposições dentro do mesmo papel são registradas, mas não são
tratadas como vazamento entre treino, validação e teste.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

VERSAO_ESQUEMA = 1
ALGORITMO_HASH = "sha256"
_PAPEIS = {
    "treino": "treino",
    "train": "treino",
    "validacao": "validacao",
    "validation": "validacao",
    "teste": "teste",
    "test": "teste",
}


class ErroManifesto(ValueError):
    """Indica especificação inválida, vazamento ou divergência dos dados."""


def _sem_acentos(valor: str) -> str:
    normalizado = unicodedata.normalize("NFKD", valor.strip().casefold())
    return "".join(caractere for caractere in normalizado if not unicodedata.combining(caractere))


def _normalizar_papel(valor: Any) -> str:
    if not isinstance(valor, str):
        raise ErroManifesto("o papel do dataset deve ser texto")
    chave = _sem_acentos(valor)
    if chave not in _PAPEIS:
        permitidos = "treino, validacao/validação ou teste"
        raise ErroManifesto(f"papel inválido {valor!r}; use {permitidos}")
    return _PAPEIS[chave]


def _texto_obrigatorio(item: Mapping[str, Any], campo: str, nome: str) -> str:
    valor = item.get(campo)
    if not isinstance(valor, str) or not valor.strip():
        raise ErroManifesto(f"dataset {nome!r}: {campo!r} deve ser texto não vazio")
    return valor.strip()


def _resolver_relativo(raiz: Path, valor: Any, descricao: str) -> tuple[Path, str]:
    if not isinstance(valor, str) or not valor.strip():
        raise ErroManifesto(f"{descricao} deve ser um caminho relativo não vazio")
    informado = Path(valor)
    if informado.is_absolute():
        raise ErroManifesto(f"{descricao} deve ser relativo à raiz: {valor!r}")
    absoluto = (raiz / informado).resolve()
    try:
        relativo = absoluto.relative_to(raiz)
    except ValueError as erro:
        raise ErroManifesto(f"{descricao} sai da raiz auditada: {valor!r}") from erro
    return absoluto, relativo.as_posix()


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _registro_arquivo(caminho: Path, raiz: Path) -> dict[str, Any]:
    try:
        relativo = caminho.resolve().relative_to(raiz).as_posix()
    except ValueError as erro:
        raise ErroManifesto(f"arquivo sai da raiz auditada: {caminho}") from erro
    return {
        "caminho": relativo,
        "sha256": _sha256(caminho),
        "tamanho_bytes": caminho.stat().st_size,
    }


def _listar_documentos(caminho: Path, raiz: Path, nome: str) -> list[dict[str, Any]]:
    if caminho.is_dir():
        arquivos = sorted(
            (item for item in caminho.rglob("*") if item.is_file() and item.suffix.casefold() == ".txt"),
            key=lambda item: item.relative_to(caminho).as_posix(),
        )
    elif caminho.is_file() and caminho.suffix.casefold() == ".txt":
        arquivos = [caminho]
    elif caminho.exists():
        raise ErroManifesto(f"dataset {nome!r}: documentos deve apontar para pasta ou arquivo .txt")
    else:
        raise ErroManifesto(f"dataset {nome!r}: caminho de documentos inexistente: {caminho}")

    if not arquivos:
        raise ErroManifesto(f"dataset {nome!r}: nenhum documento .txt encontrado")

    documentos: list[dict[str, Any]] = []
    ids: set[str] = set()
    for arquivo in arquivos:
        documento_id = arquivo.stem
        if documento_id in ids:
            raise ErroManifesto(
                f"dataset {nome!r}: documento_id duplicado dentro do conjunto: {documento_id!r}"
            )
        ids.add(documento_id)
        documentos.append({"documento_id": documento_id, **_registro_arquivo(arquivo, raiz)})
    return documentos


def _objetos_json(caminho: Path) -> Iterable[tuple[Any, str | None]]:
    sufixo = caminho.suffix.casefold()
    if sufixo == ".jsonl":
        with caminho.open(encoding="utf-8-sig") as arquivo:
            for numero, linha in enumerate(arquivo, 1):
                if linha.strip():
                    try:
                        yield json.loads(linha), None
                    except json.JSONDecodeError as erro:
                        raise ErroManifesto(f"JSONL inválido em {caminho}, linha {numero}: {erro}") from erro
        return

    try:
        conteudo = json.loads(caminho.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as erro:
        raise ErroManifesto(f"JSON inválido em {caminho}: {erro}") from erro
    if isinstance(conteudo, list):
        for item in conteudo:
            yield item, None
    elif isinstance(conteudo, dict) and isinstance(conteudo.get("anotacoes"), list):
        documento_id = conteudo.get("documento_id")
        for item in conteudo["anotacoes"]:
            yield item, documento_id if isinstance(documento_id, str) else None
    else:
        yield conteudo, None


def _expandir_objeto_anotacoes(objeto: Any, herdado: str | None) -> Iterable[tuple[Any, str | None]]:
    if isinstance(objeto, dict):
        documento_id = objeto.get("documento_id")
        if not isinstance(documento_id, str):
            documento_id = herdado
        for chave in ("anotacoes", "spans"):
            if isinstance(objeto.get(chave), list):
                for item in objeto[chave]:
                    yield item, documento_id
                return
    yield objeto, herdado


def _contar_anotacoes(caminho: Path, documentos: set[str], nome: str) -> int:
    sufixo = caminho.suffix.casefold()
    ids_referenciados: set[str] = set()
    quantidade = 0

    if sufixo == ".csv":
        with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
            leitor = csv.DictReader(arquivo)
            if leitor.fieldnames is None:
                raise ErroManifesto(f"dataset {nome!r}: CSV de anotações sem cabeçalho")
            for linha in leitor:
                quantidade += 1
                documento_id = linha.get("documento_id")
                if documento_id:
                    ids_referenciados.add(documento_id)
    elif sufixo in {".json", ".jsonl"}:
        for objeto, herdado in _objetos_json(caminho):
            for anotacao, documento_herdado in _expandir_objeto_anotacoes(objeto, herdado):
                quantidade += 1
                documento_id = anotacao.get("documento_id") if isinstance(anotacao, dict) else None
                if not isinstance(documento_id, str):
                    documento_id = documento_herdado
                if documento_id:
                    ids_referenciados.add(documento_id)
    else:
        raise ErroManifesto(
            f"dataset {nome!r}: formato de anotações não suportado {sufixo!r}; use CSV, JSON ou JSONL"
        )

    desconhecidos = sorted(ids_referenciados - documentos)
    if desconhecidos:
        amostra = ", ".join(repr(item) for item in desconhecidos[:5])
        complemento = "..." if len(desconhecidos) > 5 else ""
        raise ErroManifesto(
            f"dataset {nome!r}: anotações referenciam documentos ausentes: {amostra}{complemento}"
        )
    return quantidade


def _hash_dataset(arquivos: Sequence[Mapping[str, Any]]) -> str:
    """Combina caminho, tamanho e digest de cada arquivo de forma estável."""
    digest = hashlib.sha256()
    for arquivo in sorted(arquivos, key=lambda item: str(item["caminho"])):
        digest.update(str(arquivo["caminho"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(arquivo["tamanho_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(arquivo["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifestar_dataset(raiz: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    nome = _texto_obrigatorio(item, "nome", "<sem nome>")
    origem = _texto_obrigatorio(item, "origem", nome)
    papel = _normalizar_papel(item.get("papel"))
    caminho_documentos, relativo_documentos = _resolver_relativo(
        raiz, item.get("documentos"), f"dataset {nome!r}: documentos"
    )
    documentos = _listar_documentos(caminho_documentos, raiz, nome)

    anotacoes: dict[str, Any] | None = None
    valor_anotacoes = item.get("anotacoes")
    if valor_anotacoes is not None:
        caminho_anotacoes, _ = _resolver_relativo(
            raiz, valor_anotacoes, f"dataset {nome!r}: anotacoes"
        )
        if not caminho_anotacoes.is_file():
            raise ErroManifesto(f"dataset {nome!r}: arquivo de anotações inexistente: {caminho_anotacoes}")
        anotacoes = _registro_arquivo(caminho_anotacoes, raiz)
        anotacoes["quantidade"] = _contar_anotacoes(
            caminho_anotacoes, {item["documento_id"] for item in documentos}, nome
        )

    arquivos: list[Mapping[str, Any]] = list(documentos)
    if anotacoes is not None:
        arquivos.append(anotacoes)
    return {
        "nome": nome,
        "origem": origem,
        "papel": papel,
        "caminho_documentos": relativo_documentos,
        "caminho_anotacoes": anotacoes["caminho"] if anotacoes else None,
        "sha256": _hash_dataset(arquivos),
        "tamanho_bytes": sum(int(arquivo["tamanho_bytes"]) for arquivo in arquivos),
        "quantidade_documentos": len(documentos),
        "quantidade_anotacoes": anotacoes["quantidade"] if anotacoes else 0,
        "documentos": documentos,
        "anotacoes": anotacoes,
    }


def _sobreposicoes(datasets: Sequence[Mapping[str, Any]], campo: str) -> list[dict[str, Any]]:
    indice: dict[str, list[dict[str, str]]] = defaultdict(list)
    for dataset in datasets:
        for documento in dataset["documentos"]:
            chave = str(documento[campo])
            indice[chave].append(
                {
                    "dataset": str(dataset["nome"]),
                    "papel": str(dataset["papel"]),
                    "documento_id": str(documento["documento_id"]),
                    "caminho": str(documento["caminho"]),
                }
            )

    resultado = []
    for chave, ocorrencias in sorted(indice.items()):
        datasets_distintos = {item["dataset"] for item in ocorrencias}
        if len(datasets_distintos) < 2:
            continue
        ocorrencias.sort(key=lambda item: (item["dataset"], item["documento_id"], item["caminho"]))
        papeis = sorted({item["papel"] for item in ocorrencias})
        resultado.append(
            {
                "valor": chave,
                "papeis": papeis,
                "entre_papeis": len(papeis) > 1,
                "ocorrencias": ocorrencias,
            }
        )
    return resultado


def _auditoria(datasets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    por_id = _sobreposicoes(datasets, "documento_id")
    por_hash = _sobreposicoes(datasets, "sha256")
    vazamentos = sum(item["entre_papeis"] for item in por_id + por_hash)
    return {
        "sobreposicoes_documento_id": por_id,
        "sobreposicoes_sha256": por_hash,
        "vazamentos_entre_papeis": vazamentos,
        "aprovado": vazamentos == 0,
    }


def carregar_especificacao(caminho: Path) -> list[Mapping[str, Any]]:
    """Carrega a lista de datasets; não acessa os dados declarados ainda."""
    try:
        especificacao = json.loads(caminho.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as erro:
        raise ErroManifesto(f"especificação inexistente: {caminho}") from erro
    except json.JSONDecodeError as erro:
        raise ErroManifesto(f"especificação JSON inválida: {erro}") from erro
    if not isinstance(especificacao, dict) or especificacao.get("versao") != VERSAO_ESQUEMA:
        raise ErroManifesto(f"a especificação deve ter versao={VERSAO_ESQUEMA}")
    datasets = especificacao.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ErroManifesto("a especificação deve conter uma lista não vazia em 'datasets'")
    if not all(isinstance(item, dict) for item in datasets):
        raise ErroManifesto("cada entrada de 'datasets' deve ser um objeto JSON")
    return datasets


def gerar_manifesto(raiz: Path, datasets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gera um manifesto em memória sem criar, mover ou modificar datasets."""
    raiz = raiz.resolve()
    manifestados = [_manifestar_dataset(raiz, item) for item in datasets]
    nomes = [item["nome"] for item in manifestados]
    if len(nomes) != len(set(nomes)):
        repetidos = sorted({nome for nome in nomes if nomes.count(nome) > 1})
        raise ErroManifesto(f"nomes de dataset duplicados: {', '.join(repetidos)}")
    manifestados.sort(key=lambda item: item["nome"])
    return {
        "versao": VERSAO_ESQUEMA,
        "algoritmo_hash": ALGORITMO_HASH,
        "datasets": manifestados,
        "auditoria": _auditoria(manifestados),
    }


def gravar_manifesto(manifesto: Mapping[str, Any], caminho: Path) -> None:
    """Grava JSON estável e legível, sempre terminado por quebra de linha."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _descritores_do_manifesto(manifesto: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifesto.get("versao") != VERSAO_ESQUEMA or manifesto.get("algoritmo_hash") != ALGORITMO_HASH:
        raise ErroManifesto("versão ou algoritmo de hash do manifesto não suportado")
    datasets = manifesto.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ErroManifesto("manifesto sem datasets")
    descritores = []
    for item in datasets:
        if not isinstance(item, dict):
            raise ErroManifesto("entrada de dataset inválida no manifesto")
        descritores.append(
            {
                "nome": item.get("nome"),
                "origem": item.get("origem"),
                "papel": item.get("papel"),
                "documentos": item.get("caminho_documentos"),
                "anotacoes": item.get("caminho_anotacoes"),
            }
        )
    return descritores


def _primeira_divergencia(esperado: Any, atual: Any, caminho: str = "$") -> str | None:
    if type(esperado) is not type(atual):
        return f"{caminho}: tipo {type(esperado).__name__} != {type(atual).__name__}"
    if isinstance(esperado, dict):
        chaves_esperadas, chaves_atuais = set(esperado), set(atual)
        if chaves_esperadas != chaves_atuais:
            faltantes = sorted(chaves_esperadas - chaves_atuais)
            extras = sorted(chaves_atuais - chaves_esperadas)
            return f"{caminho}: chaves faltantes={faltantes}, extras={extras}"
        for chave in sorted(esperado):
            divergencia = _primeira_divergencia(esperado[chave], atual[chave], f"{caminho}.{chave}")
            if divergencia:
                return divergencia
        return None
    if isinstance(esperado, list):
        if len(esperado) != len(atual):
            return f"{caminho}: tamanho {len(esperado)} != {len(atual)}"
        for indice, (item_esperado, item_atual) in enumerate(zip(esperado, atual, strict=True)):
            divergencia = _primeira_divergencia(item_esperado, item_atual, f"{caminho}[{indice}]")
            if divergencia:
                return divergencia
        return None
    if esperado != atual:
        return f"{caminho}: {esperado!r} != {atual!r}"
    return None


def garantir_sem_vazamento(manifesto: Mapping[str, Any]) -> None:
    """Falha se houver ID ou hash exato compartilhado entre papéis distintos."""
    auditoria = manifesto.get("auditoria")
    if not isinstance(auditoria, dict) or auditoria.get("aprovado") is not True:
        quantidade = auditoria.get("vazamentos_entre_papeis", "desconhecida") if isinstance(auditoria, dict) else "desconhecida"
        raise ErroManifesto(
            f"vazamento entre treino/validação/teste: {quantidade} sobreposição(ões) por ID ou SHA-256"
        )


def carregar_manifesto(caminho: Path) -> dict[str, Any]:
    """Carrega um manifesto JSON e valida sua estrutura mínima."""
    try:
        manifesto = json.loads(caminho.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as erro:
        raise ErroManifesto(f"manifesto inexistente: {caminho}") from erro
    except json.JSONDecodeError as erro:
        raise ErroManifesto(f"manifesto JSON inválido: {erro}") from erro
    if not isinstance(manifesto, dict):
        raise ErroManifesto("o manifesto deve ser um objeto JSON")
    _descritores_do_manifesto(manifesto)
    return manifesto


def auditar_sobreposicao(
    avaliacao: Mapping[str, Any], treino: Mapping[str, Any]
) -> dict[str, Any]:
    """Compara dois manifestos por ID e SHA-256, mesmo que os papéis/IDs mudem.

    Esta API atende gates que mantêm o corpus de avaliação e o corpus de treino
    em manifestos separados. O resultado não depende do campo ``papel``: toda
    interseção exata entre os dois lados é considerada sobreposição.
    """

    def documentos(manifesto: Mapping[str, Any], lado: str) -> list[dict[str, str]]:
        datasets = manifesto.get("datasets")
        if not isinstance(datasets, list):
            raise ErroManifesto(f"manifesto de {lado} sem datasets")
        resultado = []
        for dataset in datasets:
            if not isinstance(dataset, dict) or not isinstance(dataset.get("documentos"), list):
                raise ErroManifesto(f"dataset inválido no manifesto de {lado}")
            for documento in dataset["documentos"]:
                if not isinstance(documento, dict):
                    raise ErroManifesto(f"documento inválido no manifesto de {lado}")
                campos = ("documento_id", "sha256", "caminho")
                if not all(isinstance(documento.get(campo), str) for campo in campos):
                    raise ErroManifesto(f"documento sem ID, SHA-256 ou caminho no manifesto de {lado}")
                resultado.append(
                    {
                        "lado": lado,
                        "dataset": str(dataset.get("nome", "")),
                        "documento_id": documento["documento_id"],
                        "sha256": documento["sha256"],
                        "caminho": documento["caminho"],
                    }
                )
        return resultado

    documentos_avaliacao = documentos(avaliacao, "avaliacao")
    documentos_treino = documentos(treino, "treino")

    def comparar(campo: str) -> list[dict[str, Any]]:
        por_avaliacao: dict[str, list[dict[str, str]]] = defaultdict(list)
        por_treino: dict[str, list[dict[str, str]]] = defaultdict(list)
        for documento in documentos_avaliacao:
            por_avaliacao[documento[campo]].append(documento)
        for documento in documentos_treino:
            por_treino[documento[campo]].append(documento)
        return [
            {
                "valor": valor,
                "ocorrencias": sorted(
                    por_avaliacao[valor] + por_treino[valor],
                    key=lambda item: (item["lado"], item["dataset"], item["documento_id"]),
                ),
            }
            for valor in sorted(set(por_avaliacao) & set(por_treino))
        ]

    por_id = comparar("documento_id")
    por_hash = comparar("sha256")
    return {
        "sobreposicoes_documento_id": por_id,
        "sobreposicoes_sha256": por_hash,
        "aprovado": not por_id and not por_hash,
    }


def auditar_manifesto(caminho: Path, raiz: Path) -> dict[str, Any]:
    """Recalcula o manifesto, falha em drift e devolve o conteúdo auditado."""
    registrado = carregar_manifesto(caminho)
    atual = gerar_manifesto(raiz, _descritores_do_manifesto(registrado))
    divergencia = _primeira_divergencia(registrado, atual)
    if divergencia:
        raise ErroManifesto(f"manifesto diverge dos dados atuais em {divergencia}")
    garantir_sem_vazamento(atual)
    return atual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    gerar = subcomandos.add_parser("gerar", help="gera o manifesto e valida sobreposições")
    gerar.add_argument("--especificacao", type=Path, required=True)
    gerar.add_argument("--saida", type=Path, required=True)
    gerar.add_argument("--raiz", type=Path, default=Path.cwd())

    auditar = subcomandos.add_parser("auditar", help="recalcula e confere um manifesto existente")
    auditar.add_argument("--manifesto", type=Path, required=True)
    auditar.add_argument("--raiz", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.comando == "gerar":
            manifesto = gerar_manifesto(args.raiz, carregar_especificacao(args.especificacao))
            gravar_manifesto(manifesto, args.saida)
            garantir_sem_vazamento(manifesto)
            print(f"Manifesto gerado e aprovado: {args.saida}")
        else:
            manifesto = auditar_manifesto(args.manifesto, args.raiz)
            print(f"Manifesto íntegro e sem vazamento: {len(manifesto['datasets'])} dataset(s)")
    except ErroManifesto as erro:
        parser.exit(1, f"erro: {erro}\n")


if __name__ == "__main__":
    main()
