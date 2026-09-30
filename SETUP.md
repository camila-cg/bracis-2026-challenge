# Setup e execução

Há três formas de rodar o pipeline: com **uv** (recomendada), com **pip** ou com **Docker**.
Todas geram o mesmo resultado. Execute os comandos na raiz do repositório.

A solução usa só regras (regex) e consultas à base SQLite: não precisa de GPU,
de modelos nem de internet durante a execução (só para instalar as dependências).

## Passo a passo: da pasta do repositório ao zip de submissão

Comandos para PowerShell (Windows). As seções seguintes detalham cada etapa.

### Sem Docker (requer git e [uv](https://docs.astral.sh/uv/getting-started/installation/))

```powershell
# 1. entrar na pasta do repositório
cd bracis-2026-challenge

# 2. copiar os dados recebidos por e-mail para data/ (ver seção 1):
#    data\txt\*.txt, data\desafio1_bracis.db e, opcionalmente, data\goldenset_offsets.csv

# 3. instalar as dependências
uv sync --no-default-groups

# 4. gerar as predições (saida\submission.csv)
uv run --no-default-groups python src/pipeline.py

# 5. validar e empacotar (saida\submissao_<commit>\)
uv run --no-default-groups python src/empacotar_submissao.py
```

### Com Docker (requer git, Docker Desktop aberto e Python 3.10+ para o passo 5)

```powershell
# 1. entrar na pasta do repositório
cd bracis-2026-challenge

# 2. copiar os dados para data/, como acima

# 3. construir a imagem (o ponto final é obrigatório)
docker build -t bracis-2026 .

# 4. gerar as predições (saida\submission.csv)
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/saida:/app/saida" bracis-2026

# 5. validar e empacotar (saida\submissao_<commit>\), fora do container
python src/empacotar_submissao.py
```

O passo 5 roda fora do container porque precisa do histórico do git, que não
entra na imagem. O script usa só a biblioteca padrão do Python, então não é
preciso instalar nenhuma dependência para ele.

### Depois do passo 5

Em `saida\submissao_<commit>\` ficam:

- `submission.csv` (ou `submission.zip`): envie no Kaggle em
  *Submit Prediction → File Upload* e escreva o commit na descrição;
- `bundle_<commit>.zip`: guarde para a verificação de reprodutibilidade.

Se houver arquivos alterados e não commitados, faça commit antes do passo 5;
senão o script recusa gerar o bundle (ver seção 5).

## 1. Dados necessários

Os dados do desafio não estão no repositório. Antes de rodar, confira se existem:

```
data/txt/                    documentos de entrada (<documento_id>.txt)
data/desafio1_bracis.db      base canônica
data/goldenset_offsets.csv   gabarito, necessário somente para --avaliar
```

`data/json_to_submission.py` e `data/kaggle_metric.py` já vêm no repositório.

## 2. Execução sem Docker

### Com uv (recomendada)

Requer o [uv](https://docs.astral.sh/uv/getting-started/installation/). Ele instala
o Python 3.12 (fixado em `.python-version`) e as versões travadas no `uv.lock`.

```bash
uv sync --no-default-groups
uv run --no-default-groups python src/pipeline.py
```

`--no-default-groups` instala só o necessário para o pipeline (`pandas`), sem os
grupos `dev` (pytest) e `notebooks` (Jupyter, matplotlib). Para desenvolver, use
`uv sync` sem a flag, que instala também esses dois grupos (definidos em
`default-groups` no `pyproject.toml`). Para rodar os testes e gates com o mínimo,
veja a seção 6.

### Com pip

Requer Python 3.10 ou superior.

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/pipeline.py
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/pipeline.py
```

## 3. Execução com Docker

Requer o Docker (no Windows, o Docker Desktop precisa estar aberto). A imagem
usa as mesmas versões do `uv.lock`. Os dados não são copiados para a imagem:
`data/` e `saida/` são montadas no container.

```bash
docker build -t bracis-2026 .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/saida:/app/saida" bracis-2026
```

No PowerShell:

```powershell
docker build -t bracis-2026 .
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/saida:/app/saida" bracis-2026
```

Argumentos depois do nome da imagem vão para o pipeline, ex.: `... bracis-2026 --avaliar`.
Só é preciso refazer o `docker build` quando o código ou as dependências mudarem.

## 4. Saídas e opções

O pipeline grava:

- `saida/json/<documento_id>.json`: um JSON por documento (contrato do desafio);
- `saida/submission.csv`: o arquivo de submissão do Kaggle;
- `outputs/extracao_<data>.csv`: um CSV de depuração da extração por execução, com os
  candidatos de cada documento. Não faz parte da submissão e fica fora do git.

Opções de `src/pipeline.py`:

| Opção | O que faz |
|---|---|
| `--avaliar` | pontua a saída contra o gabarito com a métrica oficial |
| `--txt <pasta>` | pasta dos `.txt` (padrão `data/txt`) |
| `--db <arquivo>` | base canônica (padrão `data/desafio1_bracis.db`) |
| `--saida <pasta>` | pasta de saída (padrão `saida`) |
| `--goldenset <arquivo>` | gabarito usado por `--avaliar` |

## 5. Submeter no Kaggle

Com as mudanças commitadas, gere as saídas e depois rode:

```bash
uv run python src/empacotar_submissao.py
```

O script valida o `saida/submission.csv` e cria `saida/submissao_<commit>/` com:

- `submission.csv` e `submission.zip`: envie um dos dois em *Submit Prediction → File Upload*
  e anote o commit na descrição da submissão;
- `bundle_<commit>.zip`: o código desse commit e um `comando.txt`, para a verificação de reprodutibilidade.

Se houver mudanças não commitadas, o script recusa gerar o bundle; use
`--permitir-sujo` apenas para testes.

## 6. Testes e gates

### Rodar

Instalação mínima (só `pandas` e `pytest`, o que os testes usam):

```bash
uv sync --locked --no-default-groups --group dev
uv run --locked --no-default-groups --group dev pytest -q
```

Com `uv sync --locked` sem as flags, o ambiente inclui também os notebooks, e basta
`uv run --locked pytest -q`. A suíte leva cerca de 20 s.

Use `--locked`: ele falha se o `uv.lock` não corresponder ao `pyproject.toml`, em vez
de reescrevê-lo em silêncio. Só rode `uv lock` ao mudar dependências, e commite o
`uv.lock` junto. Se aparecer `No module named pytest`, o ambiente foi criado com
`--no-default-groups`: acrescente `--group dev`.

### Dados

Os testes e gates usam `data/desafio1_bracis.db` (e `data/txt` e
`data/goldenset_offsets.csv` para `--avaliar`), que não vão no git: copie-os como na
seção 1. Sem a base, os testes que dependem dela são **pulados**, não falham. Numa
cópia limpa, sem dados, 167 testes passam e 57 são pulados; com os dados, todos
passam. O dataset sintético do holdout, `datasets/dataset_citacoes_br_sintetico_2026_seed26092217.zip`,
é versionado.

### Gates de generalização

O dev está saturado (macro-F1 1,0) e só acusa regressão. Estes gates medem fora dele;
os pisos ficam nas constantes do topo de cada teste.

| Gate | Teste | Script para ver os números |
|---|---|---|
| Registro de regressão: casos que já geraram erro e controles | `tests/test_casos_regressao.py` (lê `analise/casos_regressao.csv`) | — |
| Holdout sintético (ZIP saneado), pontuado pela métrica oficial | `tests/test_holdout_zip.py` | `uv run --locked --no-default-groups python src/validacao/holdout_zip.py --detalhes` |
| Base canônica inteira, por tribunal, com falsos positivos | `tests/test_base_inteira.py` | `uv run --locked --no-default-groups python src/validacao/teste_base_inteira.py` |
| Dev (26 documentos) | — | `uv run --locked --no-default-groups python src/pipeline.py --avaliar` |

Ao melhorar um número, suba o piso na mesma mudança. No registro de regressão, um
caso `aberto` que passa a funcionar faz o teste falhar de propósito: mude o
`modo_teste` da linha para `regressao` e atualize o `status` com o commit.

### Windows e WSL na mesma pasta

O `.venv` tem layout por sistema operacional (`Scripts` no Windows, `bin` no
Linux). Se a mesma pasta é usada nos dois, o uv recria o `.venv` a cada troca. Evite
isso apontando um ambiente por sistema, fora do repositório, por exemplo no WSL:

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/bracis-wsl
```

## 7. Notebooks

Os notebooks de análise ficam em `src/*.ipynb` (EDA) e `analise/*.ipynb` (manipulação
dos dados com pandas). Precisam do grupo `notebooks` (`ipykernel`, Jupyter, matplotlib),
que o `--no-default-groups` das seções anteriores não instala.

Com uv:

```bash
uv sync
```

Com pip, o `requirements.txt` já inclui esses pacotes.

No VS Code, selecione como kernel o Python do `.venv` do projeto. Se aparecer o aviso
de que é preciso instalar `ipykernel` e `pip`, o ambiente foi criado com
`--no-default-groups`: rode `uv sync` e reinicie o kernel. Os notebooks localizam a
raiz do projeto sozinhos, então podem ser abertos de qualquer pasta dentro do repositório.
