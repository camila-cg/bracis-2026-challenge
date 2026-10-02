"""Testes do manifesto e do gate de vazamento, somente com dados sintéticos."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from validacao import manifesto_datasets as m


def _dataset(
    raiz: Path,
    pasta: str,
    documentos: dict[str, str],
    anotacoes: list[dict[str, str]] | None = None,
) -> tuple[str, str | None]:
    diretorio = raiz / pasta / "txt"
    diretorio.mkdir(parents=True)
    for documento_id, texto in documentos.items():
        (diretorio / f"{documento_id}.txt").write_text(texto, encoding="utf-8")

    caminho_anotacoes = None
    if anotacoes is not None:
        arquivo = raiz / pasta / "goldenset_offsets.csv"
        with arquivo.open("w", encoding="utf-8", newline="") as saida:
            escritor = csv.DictWriter(saida, fieldnames=["documento_id", "inicio", "fim", "trecho"])
            escritor.writeheader()
            escritor.writerows(anotacoes)
        caminho_anotacoes = arquivo.relative_to(raiz).as_posix()
    return diretorio.relative_to(raiz).as_posix(), caminho_anotacoes


def _entrada(nome: str, documentos: str, anotacoes: str | None, papel: str) -> dict:
    return {
        "nome": nome,
        "documentos": documentos,
        "anotacoes": anotacoes,
        "origem": f"origem-{nome}",
        "papel": papel,
    }


def test_gera_manifesto_deterministico_com_caminhos_relativos_e_contagens(tmp_path: Path):
    docs, anotacoes = _dataset(
        tmp_path,
        "dev",
        {"doc_b": "segundo", "doc_a": "primeiro"},
        [
            {"documento_id": "doc_a", "inicio": "0", "fim": "3", "trecho": "pri"},
            {"documento_id": "doc_b", "inicio": "0", "fim": "3", "trecho": "seg"},
        ],
    )
    entrada = _entrada("dev", docs, anotacoes, "treino")

    primeiro = m.gerar_manifesto(tmp_path, [entrada])
    segundo = m.gerar_manifesto(tmp_path, [entrada])

    assert primeiro == segundo
    dataset = primeiro["datasets"][0]
    assert dataset["origem"] == "origem-dev"
    assert dataset["papel"] == "treino"
    assert dataset["caminho_documentos"] == "dev/txt"
    assert dataset["caminho_anotacoes"] == "dev/goldenset_offsets.csv"
    assert dataset["quantidade_documentos"] == 2
    assert dataset["quantidade_anotacoes"] == 2
    assert dataset["tamanho_bytes"] == sum(
        arquivo.stat().st_size for arquivo in (tmp_path / "dev").rglob("*") if arquivo.is_file()
    )
    assert [item["documento_id"] for item in dataset["documentos"]] == ["doc_a", "doc_b"]
    assert all(not Path(item["caminho"]).is_absolute() for item in dataset["documentos"])
    assert dataset["documentos"][0]["sha256"] == hashlib.sha256(b"primeiro").hexdigest()
    assert len(dataset["sha256"]) == 64
    assert primeiro["auditoria"]["aprovado"] is True


def test_grava_e_audita_manifesto_sem_drift(tmp_path: Path):
    docs, anotacoes = _dataset(
        tmp_path,
        "treino",
        {"doc_1": "texto"},
        [{"documento_id": "doc_1", "inicio": "0", "fim": "5", "trecho": "texto"}],
    )
    manifesto = m.gerar_manifesto(tmp_path, [_entrada("treino", docs, anotacoes, "train")])
    caminho = tmp_path / "manifesto.json"
    m.gravar_manifesto(manifesto, caminho)

    auditado = m.auditar_manifesto(caminho, tmp_path)

    assert auditado == manifesto
    assert json.loads(caminho.read_text(encoding="utf-8")) == manifesto
    assert caminho.read_bytes().endswith(b"\n")


def test_auditoria_falha_quando_documento_diverge_do_manifesto(tmp_path: Path):
    docs, _ = _dataset(tmp_path, "teste", {"doc_1": "antes"})
    manifesto = m.gerar_manifesto(tmp_path, [_entrada("teste", docs, None, "teste")])
    caminho = tmp_path / "manifesto.json"
    m.gravar_manifesto(manifesto, caminho)
    (tmp_path / "teste" / "txt" / "doc_1.txt").write_text("depois", encoding="utf-8")

    with pytest.raises(m.ErroManifesto, match="diverge dos dados atuais"):
        m.auditar_manifesto(caminho, tmp_path)


def test_detecta_mesmo_id_em_papeis_distintos_e_bloqueia(tmp_path: Path):
    treino, _ = _dataset(tmp_path, "treino", {"repetido": "conteúdo A"})
    teste, _ = _dataset(tmp_path, "teste", {"repetido": "conteúdo B"})

    manifesto = m.gerar_manifesto(
        tmp_path,
        [
            _entrada("treino", treino, None, "treino"),
            _entrada("teste", teste, None, "teste"),
        ],
    )

    sobreposicoes = manifesto["auditoria"]["sobreposicoes_documento_id"]
    assert len(sobreposicoes) == 1
    assert sobreposicoes[0]["valor"] == "repetido"
    assert sobreposicoes[0]["entre_papeis"] is True
    with pytest.raises(m.ErroManifesto, match="vazamento"):
        m.garantir_sem_vazamento(manifesto)


def test_detecta_mesmo_hash_com_ids_diferentes_em_papeis_distintos(tmp_path: Path):
    treino, _ = _dataset(tmp_path, "treino", {"id_treino": "bytes idênticos"})
    validacao, _ = _dataset(tmp_path, "validacao", {"id_validacao": "bytes idênticos"})

    manifesto = m.gerar_manifesto(
        tmp_path,
        [
            _entrada("treino", treino, None, "treino"),
            _entrada("validacao", validacao, None, "validação"),
        ],
    )

    assert manifesto["datasets"][0]["papel"] in {"treino", "validacao"}
    sobreposicoes = manifesto["auditoria"]["sobreposicoes_sha256"]
    assert len(sobreposicoes) == 1
    assert {item["documento_id"] for item in sobreposicoes[0]["ocorrencias"]} == {
        "id_treino",
        "id_validacao",
    }
    assert manifesto["auditoria"]["aprovado"] is False
    with pytest.raises(m.ErroManifesto, match="SHA-256"):
        m.garantir_sem_vazamento(manifesto)


def test_compara_manifestos_separados_por_hash_mesmo_com_ids_renomeados(tmp_path: Path):
    treino, _ = _dataset(tmp_path, "treino", {"nome_original": "conteúdo compartilhado"})
    avaliacao, _ = _dataset(tmp_path, "avaliacao", {"nome_renomeado": "conteúdo compartilhado"})
    manifesto_treino = m.gerar_manifesto(
        tmp_path, [_entrada("corpus-treino", treino, None, "treino")]
    )
    manifesto_avaliacao = m.gerar_manifesto(
        tmp_path, [_entrada("corpus-avaliacao", avaliacao, None, "teste")]
    )

    resultado = m.auditar_sobreposicao(manifesto_avaliacao, manifesto_treino)

    assert resultado["sobreposicoes_documento_id"] == []
    assert len(resultado["sobreposicoes_sha256"]) == 1
    assert resultado["aprovado"] is False


def test_sobreposicao_no_mesmo_papel_e_registrada_sem_ser_vazamento(tmp_path: Path):
    a, _ = _dataset(tmp_path, "a", {"duplicado": "mesmo"})
    b, _ = _dataset(tmp_path, "b", {"duplicado": "mesmo"})

    manifesto = m.gerar_manifesto(
        tmp_path,
        [_entrada("a", a, None, "treino"), _entrada("b", b, None, "treino")],
    )

    assert manifesto["auditoria"]["sobreposicoes_documento_id"]
    assert manifesto["auditoria"]["sobreposicoes_sha256"]
    assert manifesto["auditoria"]["vazamentos_entre_papeis"] == 0
    m.garantir_sem_vazamento(manifesto)


def test_rejeita_anotacao_que_referencia_documento_ausente(tmp_path: Path):
    docs, anotacoes = _dataset(
        tmp_path,
        "dev",
        {"presente": "texto"},
        [{"documento_id": "ausente", "inicio": "0", "fim": "1", "trecho": "t"}],
    )

    with pytest.raises(m.ErroManifesto, match="documentos ausentes"):
        m.gerar_manifesto(tmp_path, [_entrada("dev", docs, anotacoes, "treino")])


def test_rejeita_caminho_absoluto_ou_fora_da_raiz(tmp_path: Path):
    docs, _ = _dataset(tmp_path, "dev", {"doc": "texto"})
    entrada_absoluta = _entrada("dev", str((tmp_path / docs).resolve()), None, "treino")
    with pytest.raises(m.ErroManifesto, match="deve ser relativo"):
        m.gerar_manifesto(tmp_path, [entrada_absoluta])

    entrada_fora = _entrada("dev", "../fora", None, "treino")
    with pytest.raises(m.ErroManifesto, match="sai da raiz"):
        m.gerar_manifesto(tmp_path, [entrada_fora])
