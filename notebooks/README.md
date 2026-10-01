# Notebooks de análise e validação

Esta pasta reúne as análises exploratórias que orientaram a arquitetura da
solução e os notebooks usados para investigar erros e executar os principais
gates de validação.

Os notebooks complementam o pipeline, mas não são a implementação principal do
projeto. Regras de produção, normalização e resolução ficam em `src/`; resultados
exibidos nas células representam o estado do código e dos dados no momento da
última execução salva.

## Ordem recomendada

Os cinco notebooks numerados formam uma sequência de leitura. Os demais podem ser
abertos de forma independente quando os dados e as saídas necessárias já existem.

| Notebook | Objetivo |
|---|---|
| `01_eda_documentos.ipynb` | Examina os documentos de entrada, sua estrutura, os distratores, o vocabulário jurídico e o ruído do nível 2. |
| `02_eda_base_canonica.ipynb` | Explora a base SQLite, os tipos de registro, duplicatas, atos normativos, FTS e a posição dos identificadores. |
| `03_eda_gabarito.ipynb` | Valida spans e offsets do gabarito e analisa a distribuição de classes, tipos e níveis. |
| `04_analise_integrada.ipynb` | Relaciona documentos, gabarito e base canônica; constrói baselines e evidencia os trade-offs de normalização e resolução. |
| `05_indice_base_canonica.ipynb` | Analisa os parsers de cabeçalho, colisões de chave e a montagem do índice estruturado usado para resolver citações. |
| `avaliacao_erros.ipynb` | Cruza as saídas do pipeline com o gabarito para inspecionar falsos positivos, falsos negativos, classes e IDs incorretos. |
| `validacao_completa.ipynb` | Executa e consolida dev set, holdout sintético, base inteira, regressões, estresse e testes automatizados. |

## Dados necessários

As análises usam, conforme o notebook:

```text
data/
├── txt/                         # documentos de entrada
├── desafio1_bracis.db           # base canônica SQLite
└── goldenset_offsets.csv        # gabarito da amostra de desenvolvimento
```

Também podem ser necessários:

- `saida/json/` e `saida/submission.csv`, especialmente para
  `avaliacao_erros.ipynb`;
- os datasets sintéticos de `datasets/`, usados nos gates de holdout;
- `data/validation/casos_regressao.csv`, usado na validação de regressões.

Os dados restritos do desafio não são instalados pelo ambiente Python. Antes de
executar os notebooks, confirme que eles estão nas posições esperadas.

## Preparar o ambiente

Na raiz do repositório, usando PowerShell:

```powershell
uv sync
.\.venv\Scripts\python.exe -m jupyter lab notebooks
```

O comando `uv sync` instala os grupos padrão `notebooks` e `dev`, definidos em
`pyproject.toml`, incluindo JupyterLab, `ipykernel`, pandas, matplotlib e pytest.

No Jupyter ou no VS Code, selecione como kernel o Python localizado em
`.venv\Scripts\python.exe`.

## Convenções de execução

- Os notebooks localizam a raiz do repositório automaticamente e podem ser
  abertos a partir de `notebooks/` ou da raiz do projeto.
- As análises reutilizam helpers de `src/eda_utils.py` e outros módulos de
  `src/`; mudanças nesses arquivos podem alterar os resultados ao reexecutar as
  células.
- Execute os notebooks numerados na ordem para acompanhar a evolução do
  raciocínio, mas eles não compartilham estado de memória entre si.
- Reexecute todas as células antes de usar números ou gráficos em relatórios. As
  saídas armazenadas no arquivo podem estar desatualizadas em relação ao código.
- `avaliacao_erros.ipynb` pressupõe saídas recentes do pipeline. Se os arquivos
  em `saida/` não correspondem ao código atual, gere-os novamente antes da
  análise.
- `validacao_completa.ipynb` é o notebook mais demorado. Ele executa partes do
  pipeline, scripts de validação e testes, usando uma pasta temporária para as
  saídas intermediárias.

## Fonte de verdade

Use os notebooks para exploração, diagnóstico e comparação. Para reproduzir a
submissão ou verificar o comportamento vigente da solução, prefira os comandos
documentados no [`README.md`](../README.md) principal e em [`SETUP.md`](../SETUP.md).
