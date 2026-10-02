"""Contrato mínimo da imagem oficial e de sua documentação."""

import json
from pathlib import Path

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

    script = _ler("run.sh")
    assert "--modelo-citacoes /opt/models/bertimbau-large-citacoes-juridicas" in script
    assert "--falhar-em-erros" in script
    assert '--dispositivo-modelo "${BRACIS_DISPOSITIVO:-cuda}"' in script
    assert 'work_path=${BRACIS_WORK_DIR:-"$output_dir/.bracis-work"}' in script
    assert '--logs "$work_path/logs"' in script


def test_inferencia_oficial_congela_limiares_do_model_card():
    configuracao = json.loads(_ler("config/pipeline.yaml"))
    assert configuracao["limiares_modelo"] == {"LEI": 0.99, "JUR": 0.99}
    model_card = _ler("artefatos/ml_large/bertimbau/final/adaptador/README.md")
    assert "--limiar-lei 0.99 --limiar-jur 0.99" in model_card


def test_readme_destaca_docker_e_tambem_expoe_execucao_local():
    readme = _ler("README.md")
    assert "### Execução oficial: Docker híbrido" in readme
    assert "### Execução local pelo script" in readme
    assert "python src/pipeline.py --modelo-citacoes" in readme
    assert "docker run --rm --gpus all --network none" in readme
    assert "BRACIS_DISPOSITIVO=cpu" in readme
    assert "--dispositivo-modelo auto" in readme
    assert "--dispositivo-modelo cuda" in readme
    assert "--dispositivo-modelo cpu" in readme

    pipeline = _ler("src/pipeline.py")
    assert 'choices=["auto", "cuda", "cpu"]' in pipeline
    assert 'default="auto"' in pipeline


def test_model_card_usa_ambiente_ml_travado_e_documenta_dispositivos():
    model_card = _ler("artefatos/ml_large/bertimbau/final/adaptador/README.md")
    assert "requirements-ml.txt" not in model_card
    assert "uv sync --locked --no-default-groups --group ml" in model_card
    for dispositivo in ("auto", "cuda", "cpu"):
        assert f"--dispositivo-modelo {dispositivo}" in model_card


def test_smoke_cpu_declara_dispositivo_explicitamente():
    workflow = _ler(".github/workflows/docker-smoke.yml")
    assert "BRACIS_DISPOSITIVO=cpu" in workflow
