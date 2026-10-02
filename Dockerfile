# Imagem oficial de inferência: código, dependências e snapshot imutável do
# modelo são incorporados no build. A execução é inteiramente offline.
#
# Build:
#   docker build -t bracis-2026 .
# Run:
#   docker run --rm --gpus all \
#     -v /dados:/entrada:ro -v /saida:/saida \
#     bracis-2026 /entrada/desafio.db /entrada/txt /saida/submission.csv

FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUTF8=1 \
    HF_HOME=/opt/huggingface-cache

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --group ml --no-install-project

ARG HF_MODEL=giovannalavouras/bertimbau-large-citacoes-juridicas
ARG HF_REVISION=44697e852ef12493d0c9f560ad7da1e513dcdb35
RUN .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_MODEL}', revision='${HF_REVISION}', local_dir='/opt/models/bertimbau-large-citacoes-juridicas')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY . .
RUN chmod 0555 /app/run.sh

ENTRYPOINT ["/app/run.sh"]
