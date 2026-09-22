# Setup e execução

Requer Python 3.10 ou superior, acesso à internet para instalar as dependências e PowerShell. Execute os comandos na raiz do repositório.

## Dados necessários

Antes de rodar o pipeline, confira se existem:

- `data/txt/` — documentos de entrada (`.txt`);
- `data/desafio1_bracis.db` — base canônica;
- `data/goldenset_offsets.csv` — gabarito, necessário somente para `--avaliar`.

## Instalação com uv (recomendada)

```powershell
uv sync --no-default-groups
uv run --no-default-groups python src/pipeline.py --avaliar
```

Para instalar também as dependências dos notebooks, use `uv sync` sem `--no-default-groups`.

## Alternativa com pip

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/pipeline.py --avaliar
```

O pipeline grava um JSON por documento em `saida/json/` e a submissão em `saida/submission.csv`. Omita `--avaliar` para executar sem o gabarito. Use `--txt`, `--db`, `--saida` e `--goldenset` para informar caminhos diferentes.

## Testes locais

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes precisam de `data/desafio1_bracis.db`. Casos marcados como `expected failure` registram bugs conhecidos ainda não corrigidos.
