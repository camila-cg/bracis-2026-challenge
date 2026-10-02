# Setup e execução

O repositório permite desenvolvimento local com **uv** ou **pip**, mas a entrega
oficial do desafio é uma única imagem **Docker** com regras, SQLite e o detector
BERTimbau. Execute os comandos na raiz do repositório.

## Imagem Docker com BERTimbau Large (sem Ollama)

A imagem definida pelo único `Dockerfile` incorpora no build o modelo público
`giovannalavouras/bertimbau-large-citacoes-juridicas`, fixado pela revisão
`44697e852ef12493d0c9f560ad7da1e513dcdb35`. O contêiner não acessa a internet
durante a execução e não depende de Ollama. O host precisa de GPU NVIDIA, driver
compatível e NVIDIA Container Toolkit configurado.

```powershell
docker build -t bracis-2026 .

New-Item -ItemType Directory -Force saida | Out-Null
docker run --rm --gpus all --network none `
  -v "${PWD}/data:/entrada:ro" `
  -v "${PWD}/saida:/saida" `
  bracis-2026 `
  /entrada/desafio1_bracis.db /entrada/txt /saida/submission.csv
```

O download do Hugging Face ocorre apenas em `docker build`; por isso o build
precisa de internet, mas a imagem resultante é portável e executa offline.

## Passo a passo: da pasta do repositório ao zip de submissão

Comandos para PowerShell (Windows). As seções seguintes detalham cada etapa.

### Fluxo oficial com Docker e BERTimbau

```powershell
# 1. entrar na pasta do repositório
cd bracis-2026-challenge

# 2. copiar os dados recebidos por e-mail para data/ (ver seção 1):
#    data\txt\*.txt, data\desafio1_bracis.db e, opcionalmente, data\goldenset_offsets.csv

# 3. construir a imagem (requer internet; o ponto final é obrigatório)
docker build -t bracis-2026 .

# 4. gerar as predições (saida\submission.csv)
New-Item -ItemType Directory -Force saida | Out-Null
docker run --rm --gpus all --network none `
  -v "${PWD}/data:/entrada:ro" -v "${PWD}/saida:/saida" `
  bracis-2026 `
  /entrada/desafio1_bracis.db /entrada/txt /saida/submission.csv

# 5. validar e empacotar (saida\submissao_<commit>\), fora do contêiner
python src/empacotar_submissao.py
```

O passo 5 roda fora do contêiner porque precisa do histórico do Git, que não
entra na imagem. O host precisa de Git e Python 3.10+; no Windows, `py -3` pode
substituir `python`. O script usa só a biblioteca padrão, sem pacote adicional.

### Execução local pelo script

Este caminho executa a mesma composição de detector, regras e SQLite, mas o
Docker acima continua sendo o formato de entrega da competição.

```powershell
$modelo = ".hf-cache\bertimbau-large-citacoes-juridicas"
uv sync --locked --no-default-groups --group ml
uv run --locked --no-default-groups --group ml --no-sync hf download giovannalavouras/bertimbau-large-citacoes-juridicas --revision 44697e852ef12493d0c9f560ad7da1e513dcdb35 --local-dir $modelo
uv run --locked --no-default-groups --group ml --no-sync python src/pipeline.py --modelo-citacoes $modelo
python src/empacotar_submissao.py
```

Acrescente `--avaliar` ao comando de `src/pipeline.py` para pontuar o fluxo
híbrido localmente. O entrypoint Docker aceita somente os três argumentos
posicionais e não executa a avaliação.

Para a baseline leve de regras, sem o detector neural, use somente:

```powershell
uv sync --locked --no-default-groups
uv run --locked --no-default-groups --no-sync python src/pipeline.py
```

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

## 2. Execução local pelo script

### Fluxo híbrido com uv

Requer o [uv](https://docs.astral.sh/uv/getting-started/installation/). Ele instala
o Python 3.12 (fixado em `.python-version`) e as versões travadas no `uv.lock`.

```bash
MODELO_LOCAL=".hf-cache/bertimbau-large-citacoes-juridicas"
uv sync --locked --no-default-groups --group ml
uv run --locked --no-default-groups --group ml --no-sync hf download giovannalavouras/bertimbau-large-citacoes-juridicas --revision 44697e852ef12493d0c9f560ad7da1e513dcdb35 --local-dir "$MODELO_LOCAL"
uv run --locked --no-default-groups --group ml --no-sync python src/pipeline.py --modelo-citacoes "$MODELO_LOCAL"
```

O download do modelo exige internet apenas na primeira execução. Depois disso,
o caminho em `.hf-cache` é local. O detector usa a GPU quando CUDA está
disponível e recua para CPU caso contrário, com desempenho consideravelmente
menor.

### Baseline de regras

```bash
uv sync --locked --no-default-groups
uv run --locked --no-default-groups --no-sync python src/pipeline.py
```

Nesse caso, `--no-default-groups` instala só o necessário para as regras
(`pandas`). A baseline não carrega o BERTimbau e não equivale à entrega oficial.
Para desenvolver também com `pytest` e notebooks, use `uv sync` sem a flag; os
grupos `dev` e `notebooks` estão em `default-groups` no `pyproject.toml`.

### Com pip — baseline de regras

Requer Python 3.10 ou superior.
As versões em `requirements.txt` não são totalmente travadas; use `uv` quando
precisar de um ambiente auditável.

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

## 3. Execução oficial com Docker

Requer Docker, GPU NVIDIA e o NVIDIA Container Toolkit (no Windows, Docker
Desktop com backend WSL2). A imagem usa as versões do `uv.lock`. Os dados não
são copiados para a imagem: as pastas de entrada e saída são montadas no
contêiner.

```bash
docker build -t bracis-2026 .
mkdir -p saida
docker run --rm --gpus all --network none \
  -v "$PWD/data:/entrada:ro" -v "$PWD/saida:/saida" \
  bracis-2026 \
  /entrada/desafio1_bracis.db /entrada/txt /saida/submission.csv
```

No PowerShell:

```powershell
docker build -t bracis-2026 .
New-Item -ItemType Directory -Force saida | Out-Null
docker run --rm --gpus all --network none `
  -v "${PWD}/data:/entrada:ro" -v "${PWD}/saida:/saida" `
  bracis-2026 `
  /entrada/desafio1_bracis.db /entrada/txt /saida/submission.csv
```

A entrada recebe exatamente três argumentos posicionais: banco SQLite, pasta de
TXT e arquivo CSV final. Para persistir depois de `--rm`, o CSV precisa estar
dentro do volume de saída montado, como `/saida/submission.csv` no exemplo.
Só é preciso refazer o `docker build` quando o código ou as dependências mudarem.
O modelo usado no build é
`giovannalavouras/bertimbau-large-citacoes-juridicas`, na revisão imutável
`44697e852ef12493d0c9f560ad7da1e513dcdb35`. O download ocorre apenas no build;
o contêiner executa com acesso do Hugging Face desativado e pode ser iniciado
com `--network none` depois de a imagem estar pronta.

## 4. Saídas e opções

Na execução local, o pipeline grava:

- `saida/json/<documento_id>.json`: um JSON por documento (contrato do desafio);
- `saida/submission.csv`: o arquivo de submissão do Kaggle;
- `outputs/extracao_<data>.csv`: um CSV de depuração da extração por execução, com os
  candidatos de cada documento. Não faz parte da submissão e fica fora do git.

No contêiner oficial, o terceiro argumento é o único artefato persistido pelo
comando documentado. Os JSONs, logs e diagnósticos ficam no filesystem temporário
do contêiner; execute o pipeline localmente se precisar inspecioná-los.

Opções de `src/pipeline.py`:

| Opção | O que faz |
|---|---|
| `--avaliar` | pontua a saída contra o gabarito com a métrica oficial |
| `--txt <pasta>` | pasta dos `.txt` (padrão `data/txt`) |
| `--db <arquivo>` | base canônica (padrão `data/desafio1_bracis.db`) |
| `--saida <pasta>` | pasta de saída (padrão `saida`) |
| `--arquivo-saida <arquivo>` | caminho exato do CSV final; usado pelo wrapper Docker |
| `--logs <pasta>` | pasta dos registros JSONL de erros recuperáveis |
| `--goldenset <arquivo>` | gabarito usado por `--avaliar` |
| `--modelo-citacoes <pasta>` | ativa um artefato local do detector BIO |

## 5. Submeter no Kaggle

Com as mudanças commitadas, gere as saídas e depois rode:

```bash
python src/empacotar_submissao.py
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
uv run --locked --no-default-groups --group dev --no-sync pytest -q
```

Com `uv sync --locked` sem as flags, o ambiente inclui também os notebooks, e basta
`uv run --locked pytest -q`. A suíte leva cerca de 1 min; o gate do estresse
(`tests/test_estresse.py`) responde por cerca de 35 s disso.

Use `--locked`: ele falha se o `uv.lock` não corresponder ao `pyproject.toml`, em vez
de reescrevê-lo em silêncio. Só rode `uv lock` ao mudar dependências, e commite o
`uv.lock` junto. Se aparecer `No module named pytest`, o ambiente foi criado com
`--no-default-groups`: acrescente `--group dev`.

### Dados

Os testes e gates usam `data/desafio1_bracis.db` (e `data/txt` e
`data/goldenset_offsets.csv` para `--avaliar`), que não vão no git: copie-os como na
seção 1. Sem a base, os testes que dependem dela são **pulados**, não falham; a
quantidade exata varia à medida que a suíte evolui. O dataset sintético do
holdout, `datasets/dataset_citacoes_br_sintetico_2026_seed26092217.zip`, é
versionado.

### Gates de generalização

O dev está saturado (macro-F1 1,0) e só acusa regressão. Estes gates medem fora dele;
os pisos ficam nas constantes do topo de cada teste.

| Gate | Teste | Script para ver os números |
|---|---|---|
| Registro de regressão: casos que já geraram erro e controles | `tests/test_casos_regressao.py` (lê `data/validation/casos_regressao.csv`) | — |
| Holdout sintético (ZIP saneado), pontuado pela métrica oficial | `tests/test_holdout_zip.py` | `uv run --locked --no-default-groups python src/validacao/holdout_zip.py --detalhes` |
| Base canônica inteira, por tribunal, com falsos positivos | `tests/test_base_inteira.py` | `uv run --locked --no-default-groups python src/validacao/teste_base_inteira.py` |
| Estresse: ruído e troca de dígito nas citações do dev (4 execuções, cerca de 35 s) | `tests/test_estresse.py` | `uv run --locked --no-default-groups python src/validacao/estresse.py --seeds 3` (cerca de 1 min) |
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

Os notebooks de análise ficam em `notebooks/*.ipynb`. Precisam do grupo `notebooks`
(`ipykernel`, Jupyter, matplotlib),
que o `--no-default-groups` das seções anteriores não instala.

Com uv:

```bash
uv sync --locked
```

O `requirements.txt` é apenas o fallback mínimo do pipeline padrão e não inclui
as ferramentas de notebooks. Para trabalhar com os notebooks, use o fluxo com
`uv`, que mantém as versões no `uv.lock`.

No VS Code, selecione como kernel o Python do `.venv` do projeto. Se aparecer o aviso
de que é preciso instalar `ipykernel` e `pip`, o ambiente foi criado com
`--no-default-groups`: rode `uv sync --locked` e reinicie o kernel. Os notebooks localizam a
raiz do projeto sozinhos, então podem ser abertos de qualquer pasta dentro do repositório.

## 8. Extensões opcionais de ML e LLM

As dependências experimentais são mantidas como grupos no `pyproject.toml`; não
há arquivos `requirements` separados para elas. Para treinar ou executar o
detector BIO/BERTimbau, instale o grupo `ml`:

```bash
uv sync --group ml
```

Para usar o verificador Llama local em 4 bits junto com o detector, instale os
dois grupos. Esse fluxo requer GPU NVIDIA compatível com CUDA:

```bash
uv sync --group ml --group llm
```

O backend Ollama não precisa do grupo `llm`, pois o modelo roda no processo do
Ollama. Ele ainda precisa do grupo `ml` quando o pipeline usa o detector BIO via
`--modelo-citacoes`.
