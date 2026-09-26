# Ambiente de reprodução: mesmas versões do uv.lock, sem os grupos de notebooks/dev.
# Os dados não entram na imagem (distribuição restrita): monte data/ e saida/ no run.
#
#   docker build -t bracis-2026 .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/saida:/app/saida" bracis-2026
#
# Argumentos extras vão para o pipeline, ex.: ... bracis-2026 --avaliar
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never PYTHONUTF8=1

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY . .

ENTRYPOINT ["uv", "run", "--locked", "--no-default-groups", "--no-sync", "python", "src/pipeline.py"]
