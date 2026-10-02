"""Contrato mínimo da imagem oficial e de sua documentação."""

from pathlib import Path

from empacotar_submissao import COMANDO


RAIZ = Path(__file__).resolve().parents[1]
MODELO = "giovannalavouras/bertimbau-large-citacoes-juridicas"
REVISAO = "44697e852ef12493d0c9f560ad7da1e513dcdb35"


def _ler(nome: str) -> str:
    return (RAIZ / nome).read_text(encoding="utf-8")


def test_imagem_copia_somente_o_runtime_allowlisted():
    dockerfile = _ler("Dockerfile")
    dockerignore = _ler(".dockerignore").splitlines()

    assert "*" in dockerignore
    assert "COPY . ." not in dockerfile
    for caminho in (
        "COPY config/ ./config/",
        "COPY src/ ./src/",
        "COPY data/json_to_submission.py ./data/json_to_submission.py",
        "COPY run.sh ./run.sh",
    ):
        assert caminho in dockerfile


def test_modelo_e_revisao_sao_os_mesmos_em_todos_os_pontos_de_entrada():
    for nome in ("Dockerfile", "README.md", "SETUP.md"):
        conteudo = _ler(nome)
        assert MODELO in conteudo
        assert REVISAO in conteudo

    assert "--modelo-citacoes /opt/models/bertimbau-large-citacoes-juridicas" in _ler("run.sh")


def test_bundle_documenta_o_docker_oficial_offline():
    assert COMANDO.startswith("docker build -t bracis-2026 .")
    assert "docker run --rm --gpus all --network none" in COMANDO
    assert "/saida/submission.csv" in COMANDO


def test_readme_destaca_docker_e_tambem_expoe_execucao_local():
    readme = _ler("README.md")
    assert "### Execução oficial: Docker híbrido" in readme
    assert "### Execução local pelo script" in readme
    assert "python src/pipeline.py --modelo-citacoes" in readme
