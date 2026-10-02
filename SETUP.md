# Setup e execução

O repositório permite desenvolvimento local com **uv** ou **pip**, mas a entrega
oficial do desafio é uma única imagem **Docker** com regras, SQLite e o detector
BERTimbau. Execute os comandos na raiz do repositório.

## Imagem Docker com BERTimbau Large

A imagem definida pelo único `Dockerfile` incorpora no build o modelo público
`giovannalavouras/bertimbau-large-citacoes-juridicas`, fixado pela revisão
`44697e852ef12493d0c9f560ad7da1e513dcdb35`. O contêiner não acessa a internet
durante a execução. **GPU NVIDIA é o padrão** e requer driver compatível e
NVIDIA Container Toolkit; CPU é uma alternativa explícita.

```powershell
docker build -t bracis-2026 .

New-Item -ItemType Directory -Force saida | Out-Null
docker run --rm --gpus all --network none `
  -v "${PWD}/data:/entrada:ro" `
  -v "${PWD}/saida:/saida" `
  bracis-2026 `
  /entrada/desafio1_bracis.db /entrada/txt /saida/submission.csv
```

O comando acima é o fluxo padrão e exige CUDA. Para usar CPU, omita `--gpus all`
e acrescente `-e BRACIS_DISPOSITIVO=cpu` ao comando `docker run`.

O download do Hugging Face ocorre apenas em `docker build`; por isso o build
precisa de internet, mas a imagem resultante é portável e executa offline.

## Passo a passo: da pasta do repositório ao CSV final

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

```

### Execução local pelo script

Este caminho executa a mesma composição de detector, regras e SQLite, mas o
Docker acima continua sendo o formato de entrega da competição.

```powershell
$modelo = ".hf-cache\bertimbau-large-citacoes-juridicas"
uv sync --locked --no-default-groups --group ml
uv run --locked --no-default-groups --group ml --no-sync hf download giovannalavouras/bertimbau-large-citacoes-juridicas --revision 44697e852ef12493d0c9f560ad7da1e513dcdb35 --local-dir $modelo
uv run --locked --no-default-groups --group ml --no-sync python src/pipeline.py --modelo-citacoes $modelo --dispositivo-modelo auto
```

Localmente, `auto` prefere CUDA e recua para CPU. Use
`--dispositivo-modelo cuda` para exigir GPU ou `--dispositivo-modelo cpu` para
forçar a execução sem GPU.

Acrescente `--avaliar` ao comando de `src/pipeline.py` para pontuar o fluxo
híbrido localmente. O entrypoint Docker aceita somente os três argumentos
posicionais e não executa a avaliação.

Para a baseline leve de regras, sem o detector neural, use somente:

```powershell
uv sync --locked --no-default-groups
uv run --locked --no-default-groups --no-sync python src/pipeline.py
```

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
uv run --locked --no-default-groups --group ml --no-sync python src/pipeline.py --modelo-citacoes "$MODELO_LOCAL" --dispositivo-modelo auto
```

O download do modelo exige internet apenas na primeira execução. Depois disso,
o caminho em `.hf-cache` é local. `auto` usa CUDA quando disponível e recua para
CPU. Para tornar a escolha estrita, use `--dispositivo-modelo cuda` ou
`--dispositivo-modelo cpu`.

### Baseline de regras

```bash
uv sync --locked --no-default-groups
uv run --locked --no-default-groups --no-sync python src/pipeline.py
```

Nesse caso, `--no-default-groups` instala só o necessário para as regras
(`pandas`). A baseline não carrega o BERTimbau e não equivale à entrega oficial.
Para desenvolver também com `pytest` e notebooks, use `uv sync --locked` sem
`--no-default-groups`; os grupos `dev` e `notebooks` estão em `default-groups`
no `pyproject.toml`.

### Com pip — ambiente completo

Requer Python 3.10 ou superior.
O `requirements.txt` reúne pipeline, testes, notebooks e extensões de ML/LLM.
As versões não são totalmente travadas e os índices CUDA por plataforma não são
reproduzidos; use `uv` quando precisar de um ambiente auditável ou da seleção
correta da distribuição do PyTorch.

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

Requer Docker; no Windows, use o Docker Desktop com backend WSL2. O padrão usa
GPU NVIDIA, driver compatível e NVIDIA Container Toolkit. A imagem usa as versões do `uv.lock`. Os dados não são
copiados para a imagem: as pastas de entrada e saída são montadas no contêiner.

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

Esses comandos executam no modo padrão, com GPU. Em hosts sem GPU, omita
`--gpus all` e acrescente `-e BRACIS_DISPOSITIVO=cpu`; apenas remover a flag de
GPU não basta, pois o wrapper exige CUDA por padrão.

Exemplo em CPU (Bash):

```bash
docker run --rm --network none -e BRACIS_DISPOSITIVO=cpu \
  -v "$PWD/data:/entrada:ro" -v "$PWD/saida:/saida" \
  bracis-2026 \
  /entrada/desafio1_bracis.db /entrada/txt /saida/submission.csv
```

A entrada recebe exatamente três argumentos posicionais: banco SQLite, pasta de
TXT e arquivo CSV final. Para persistir depois de `--rm`, o CSV precisa estar
dentro do volume de saída montado, como `/saida/submission.csv` no exemplo.
O wrapper opera em modo estrito: se o detector, um documento ou uma citação
falhar, o processo termina com código diferente de zero e não grava uma
submissão parcial. O diagnóstico fica no volume de saída, em
`/saida/.bracis-work/logs/`.
Só é preciso refazer o `docker build` quando o código ou as dependências mudarem.
O modelo usado no build é
`giovannalavouras/bertimbau-large-citacoes-juridicas`, na revisão imutável
`44697e852ef12493d0c9f560ad7da1e513dcdb35`. O download ocorre apenas no build;
o contêiner executa com acesso do Hugging Face desativado e pode ser iniciado
com `--network none` depois de a imagem estar pronta.

### Se o Docker Desktop não iniciar no Windows

O erro `dockerDesktopLinuxEngine: The system cannot find the file specified`
indica que o cliente está instalado, mas o engine Linux não está disponível.
Executar em CPU elimina a exigência de GPU, porém **não** elimina a necessidade
do backend WSL2/virtualização usado pelo Docker Desktop. Confirme:

```powershell
docker info
wsl --status
Get-ComputerInfo -Property HyperVisorPresent,HyperVRequirementVirtualizationFirmwareEnabled
```

Se o WSL informar que a virtualização não está habilitada, ative a virtualização
de hardware no firmware/BIOS, habilite os componentes WSL2 e Plataforma de
Máquina Virtual do Windows e reinicie o host. Só prossiga quando `docker info`
mostrar a seção `Server`; trocar `--gpus all` por CPU não corrige um engine
inativo.

## 4. Saídas e opções

Na execução local, o pipeline grava:

- `saida/json/<documento_id>.json`: um JSON por documento (contrato do desafio);
- `saida/submission.csv`: o CSV final produzido pelo pipeline;
- `outputs/extracao_<data>.csv`: um CSV de depuração da extração por execução, com os
  candidatos de cada documento. Não faz parte da submissão e fica fora do git.

No contêiner oficial, o terceiro argumento é o CSV final. O mesmo volume recebe
`.bracis-work/json/`, com os JSONs por documento, e
`.bracis-work/logs/`, criado quando ocorre uma falha. O CSV de depuração da
extração continua temporário e não faz parte da submissão.

Opções de `src/pipeline.py`:

| Opção | O que faz |
|---|---|
| `--avaliar` | pontua a saída contra o gabarito com a métrica oficial |
| `--txt <pasta>` | pasta dos `.txt` (padrão `data/txt`) |
| `--db <arquivo>` | base canônica (padrão `data/desafio1_bracis.db`) |
| `--saida <pasta>` | pasta de saída (padrão `saida`) |
| `--arquivo-saida <arquivo>` | caminho exato do CSV final; usado pelo wrapper Docker |
| `--logs <pasta>` | pasta dos registros JSONL de erros recuperáveis |
| `--falhar-em-erros` | não gera CSV se houver qualquer erro recuperável; usado pelo Docker oficial |
| `--goldenset <arquivo>` | gabarito usado por `--avaliar` |
| `--modelo-citacoes <pasta>` | ativa um artefato local do detector BIO |
| `--dispositivo-modelo <auto|cuda|cpu>` | seleciona o dispositivo; local usa `auto` por padrão e Docker usa `cuda` |

## 5. Gerar um ZIP opcional

O CSV já é a saída final. Se outro processo exigir um ZIP, rode:

```bash
python src/empacotar_submissao.py
```

O utilitário cria apenas `saida/submission.zip` a partir do CSV. Ele é opcional,
não faz parte da execução oficial e não
empacota código, histórico Git nem comandos de entrega.

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
quantidade exata varia à medida que a suíte evolui. O dataset de regressão
sintética, `datasets/dataset_citacoes_br_sintetico_2026_seed26092217.zip`, é
versionado.

### Gates de generalização

O dev está saturado (macro-F1 1,0) e só acusa regressão. Os gates abaixo medem
robustez das regras e cobertura da base; os pisos ficam nas constantes do topo
de cada teste. O ZIP sintético não é um holdout independente do detector neural:
seus 26 documentos constam no manifesto de treino do artefato final. Portanto,
ele não pode sustentar afirmações de generalização do modelo ou do híbrido.

| Gate | Teste | Script para ver os números |
|---|---|---|
| Registro de regressão: casos que já geraram erro e controles | `tests/test_casos_regressao.py` (lê `data/validation/casos_regressao.csv`) | — |
| Regressão sintética (ZIP saneado; sobreposta ao treino neural), pontuada pela métrica oficial | `tests/test_holdout_zip.py` | `uv run --locked --no-default-groups python src/validacao/holdout_zip.py --detalhes` |
| Base canônica inteira, por tribunal, com falsos positivos | `tests/test_base_inteira.py` | `uv run --locked --no-default-groups python src/validacao/teste_base_inteira.py` |
| Estresse: ruído e troca de dígito nas citações do dev (4 execuções, cerca de 35 s) | `tests/test_estresse.py` | `uv run --locked --no-default-groups python src/validacao/estresse.py --seeds 3` (cerca de 1 min) |
| Dev (26 documentos) | — | `uv run --locked --no-default-groups python src/pipeline.py --avaliar` |

Para o detector e a fusão, use `manifesto_datasets.py` para registrar treino e
avaliação e depois `gate_hibrido.py`. Esse gate recusa sobreposição por ID ou
SHA-256 antes de carregar o modelo, executa ablações pareadas de regras/modelo/
fusão e usa a métrica oficial completa. O passo a passo e o comando com os
limiares oficiais `0.99/0.99` estão em
[`src/validacao/README.md`](src/validacao/README.md). Ele só pode ser concluído
quando houver um holdout independente; não reutilize o ZIP sintético atual.

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

Com `pip`, o `requirements.txt` completo também inclui essas ferramentas. O fluxo
com `uv` é recomendado porque mantém as versões no `uv.lock`.

No VS Code, selecione como kernel o Python do `.venv` do projeto. Se aparecer o aviso
de que é preciso instalar `ipykernel` e `pip`, o ambiente foi criado com
`--no-default-groups`: rode `uv sync --locked` e reinicie o kernel. Os notebooks localizam a
raiz do projeto sozinhos, então podem ser abertos de qualquer pasta dentro do repositório.

## 8. Extensões opcionais de ML e LLM

As dependências experimentais são mantidas como grupos no `pyproject.toml` e
também consolidadas no `requirements.txt`; não há arquivos `requirements`
separados para elas. Com `uv`, instale somente o grupo necessário. Para treinar
ou executar o detector BIO/BERTimbau:

```bash
uv sync --group ml
```

Para usar o verificador Llama local em 4 bits junto com o detector, instale os
dois grupos. Esse fluxo requer GPU NVIDIA compatível com CUDA:

```bash
uv sync --group ml --group llm
```
