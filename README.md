# Desafio Jusbrasil × BRACIS 2026

Solução para identificar citações de jurisprudência e legislação em documentos
jurídicos e classificá-las como `real`, `inventada` ou `incompleta`.

> A descrição completa da tarefa, da métrica e das regras da competição está em
> [docs/descricao_desafio.md](docs/descricao_desafio.md).

## Sobre o projeto

O desafio propõe a construção de um verificador automático de fontes jurídicas.
A partir de documentos em texto, a solução precisa localizar cada citação,
preservar seus offsets e verificar se os identificadores informados correspondem
a um registro da base canônica.

Este repositório implementa um pipeline determinístico, auditável e executável
offline. O fluxo principal combina regras de extração e normalização com
consultas a uma base SQLite. Com a mesma entrada e a mesma configuração, a saída
gerada é sempre a mesma.

## Como a solução funciona

O pipeline separa a detecção textual da verificação jurídica. Primeiro encontra
trechos que podem representar citações; depois estrutura os identificadores de
cada trecho e consulta a base canônica para decidir a classificação.

```text
documentos .txt
      ↓
extração de candidatos
      ↓
filtro de metadados e distratores
      ↓
roteamento por família jurídica
      ↓
normalização sem alterar os offsets
      ↓
consulta e desambiguação na base SQLite
      ↓
classificação + confiança
      ↓
JSONs de diagnóstico + submission.csv
```

### 1. Extração de candidatos

O texto é tokenizado e analisado em busca de marcadores jurídicos e indícios
próximos, como classe processual, número, tribunal, UF, artigo, diploma legal,
relatoria e ano. Esses registros são agrupados e pontuados para formar spans de
citações de jurisprudência e legislação.

Os vocabulários reconhecem abreviações, flexões e parte do ruído encontrado nos
documentos de nível 2, incluindo trocas de caracteres provocadas por OCR.

Internamente, a etapa segue estes subpassos:

1. **Tokenização:** cria tokens ligados às posições do texto original e mantém
   uma forma normalizada para as comparações.
2. **Detecção de evidências:** procura marcadores principais e indícios
   complementares, como números, tribunais, UFs, datas, relatores e órgãos
   julgadores.
3. **Agrupamento:** reúne evidências próximas, respeitando limites de distância,
   tamanho, sentença e mudança de fonte jurídica.
4. **Validação do grupo:** define se o conjunto representa legislação ou
   jurisprudência e verifica se há evidência mínima para formar uma citação.
5. **Pontuação e recorte:** soma as evidências sem duplicá-las, calcula o span e
   mantém apenas candidatos que atingem o limiar configurado.

### 2. Filtro contextual

Antes da classificação, o pipeline remove candidatos que parecem citações por
conter números, mas pertencem a outros contextos. Entre os casos tratados estão:

- número do processo principal e demais campos do cabeçalho;
- valor da causa, folhas dos autos, OAB e protocolo;
- metadados rotulados no início do documento.

O filtro identifica a transição entre cabeçalho e corpo para preservar uma
referência legítima feita na fundamentação, mesmo quando o cabeçalho é longo.

Internamente, a etapa segue estes subpassos:

1. **Leitura estrutural do início do documento:** percorre as linhas para
   reconhecer rótulos como `Processo`, `Autos`, `Relator` e `Valor da causa`.
2. **Identificação do processo principal:** normaliza os números encontrados nos
   campos de processo e determina quando existe um identificador principal
   inequívoco.
3. **Detecção do início do corpo:** procura títulos jurídicos, como `EMENTA`,
   `RELATÓRIO` e `VOTO`, ou a primeira linha com estrutura de texto corrido.
4. **Análise do candidato:** verifica distratores lexicais, valores monetários e
   sobreposição com os campos de metadados detectados.
5. **Decisão conservadora:** exclui somente candidatos com evidência contextual
   suficiente; os demais seguem no pipeline, acompanhados do trecho original.

### 3. Roteamento e normalização

Cada candidato é encaminhado para uma das famílias suportadas: lei, súmula,
tema, citação descritiva ou acórdão. A solução então extrai os atributos
relevantes e normaliza variações como:

- números processuais no padrão CNJ e formatos antigos;
- pontos, espaços, hífens, travessões e quebras de linha dentro do número;
- caracteres confundidos pelo OCR, como `O`/`0`, `l`/`1` e `S`/`5`;
- siglas de tribunais, UFs, classes processuais e modificadores recursais;
- diplomas legais, apelidos de leis, artigos e listas de artigos;
- súmulas, súmulas vinculantes, temas e enunciados.

O trecho original e seus offsets são preservados. As formas normalizadas servem
somente para a consulta e para a decisão semântica.

Internamente, a etapa segue estes subpassos:

1. **Escolha da rota:** usa o tipo detectado e marcadores como `lei`, `súmula` e
   `tema`; citações numeradas seguem como acórdão e referências sem número podem
   seguir como descritivas.
2. **Coleta de contexto local:** adiciona pequenas janelas à esquerda e à
   direita do span para interpretar qualificadores próximos sem ampliar a
   citação devolvida.
3. **Normalização textual:** uniformiza Unicode, separadores numéricos e ruídos
   de OCR sem reescrever o documento de entrada.
4. **Extração semântica:** seleciona número, tribunal, classe, UF, diploma,
   artigo e demais atributos aplicáveis à família escolhida.
5. **Criação do contrato intermediário:** produz uma `CitacaoNormalizada` que
   separa o texto original dos campos usados na consulta.

### 4. Resolução na base canônica

Ao iniciar, o `Resolvedor` cria índices em memória a partir da base SQLite. As
consultas combinam os identificadores disponíveis — como tribunal, classe,
número, UF, origem, diploma e artigo — para localizar registros compatíveis.

Para acórdãos, a solução tenta primeiro o índice estruturado. Se ele não produz
um resultado, usa uma busca por número como fallback e valida os metadados dos
registros encontrados antes de aceitá-los. Isso evita que uma simples
coincidência numérica seja marcada como uma citação verificada.

Internamente, a etapa segue estes subpassos:

1. **Preparação dos índices:** carrega acórdãos e atos normativos da base e cria
   mapas por número, identidade jurídica e metadados verificáveis.
2. **Consulta específica por família:** leis são buscadas por diploma e artigo;
   súmulas, por tribunal e número; acórdãos, por número e qualificadores.
3. **Validação de identidade:** rejeita qualificadores incompatíveis, números
   malformados e combinações que contradizem os dados da base.
4. **Fallback controlado:** quando o índice estruturado de acórdãos não resolve,
   busca pelo número e reaplica os filtros de tribunal, classe e UF.
5. **Desambiguação:** usa modificadores recursais e a classe citada somente
   quando esses sinais deixam um único registro; caso contrário, preserva todos
   os candidatos possíveis.
6. **Resultado auditável:** devolve um `ResultadoResolucao` com classe, ID,
   caminho de decisão, candidatos, cobertura e motivo.

### 5. Classificação e confiança

A classe representa o resultado da resolução, não apenas a presença de um padrão
no texto:

| Resultado da resolução | Classificação | Saída |
|---|---|---|
| Um único registro compatível | `real` | inclui o `id_canonico` encontrado |
| Identificadores suficientes, mas nenhum registro compatível | `inventada` | não inclui `id_canonico` |
| Identificadores ausentes, insuficientes ou compatíveis com vários registros | `incompleta` | preserva a ambiguidade para revisão |

Há regras específicas por família. Por exemplo, uma referência a lei, decreto ou
resolução sem artigo é `incompleta`, porque a base registra dispositivos e não o
diploma inteiro. Números explicitamente inválidos ou atos numerados sem
correspondência na cobertura do desafio são tratados como `inventada`.

Além da classe, o JSON de diagnóstico informa o caminho de decisão, a cobertura
da base e o motivo do resultado. A confiança é atribuída de acordo com esse
caminho e pode ser ajustada em `config/pipeline.yaml`.

Internamente, a etapa segue estes subpassos:

1. **Avaliação da evidência:** considera a suficiência dos identificadores e a
   quantidade de registros compatíveis retornados pelo resolvedor.
2. **Aplicação das regras da família:** trata situações como diploma sem artigo,
   súmula sem tribunal, tema não indexado e citação meramente descritiva.
3. **Proteção do contrato:** permite `id_canonico` somente para citações `real` e
   exige que toda citação `real` tenha exatamente esse identificador.
4. **Registro do caminho:** associa cobertura e motivo à decisão para distinguir
   ausência na base, consulta não realizada e ambiguidade entre registros.
5. **Atribuição de confiança:** seleciona o valor configurado para o caminho de
   decisão e garante que o resultado permaneça no intervalo entre `0` e `1`.

### 6. Geração das saídas

Depois que todas as citações de um documento foram processadas, o pipeline
consolida os resultados e gera os artefatos usados na análise e na submissão.

Internamente, a etapa segue estes subpassos:

1. **Montagem do documento:** agrupa as citações processadas sob o respectivo
   `documento_id`.
2. **Diagnóstico da extração:** registra em CSV os candidatos encontrados antes
   dos filtros para facilitar a investigação de falsos positivos e negativos.
3. **Persistência detalhada:** grava um JSON por documento, mantendo trechos,
   offsets, classificação, resolução, confiança e dados de auditoria.
4. **Conversão oficial:** usa `data/json_to_submission.py` para codificar as
   citações no formato esperado pelo desafio.
5. **Avaliação opcional:** com `--avaliar`, executa a métrica oficial somente
   depois que as predições já foram produzidas.

## Reprodução e submissão

A execução padrão usa somente regras e consultas à base canônica SQLite. Não
carrega modelos de linguagem, não requer GPU nem acessa a internet durante o
processamento, e a saída é determinística: a mesma entrada gera sempre o mesmo
`submission.csv`.

### Ambiente

- Python 3.12 (fixado em `.python-version`; funciona a partir de 3.10);
- [uv](https://docs.astral.sh/uv/) com `pyproject.toml` + `uv.lock` (versões travadas);
  sem uv, use `requirements.txt` (ver [SETUP.md](SETUP.md)).

### Dados

Os dados do desafio não estão no repositório (distribuição restrita). Coloque-os assim:

```
data/txt/                    documentos de entrada (<documento_id>.txt)
data/desafio1_bracis.db      base canônica
data/goldenset_offsets.csv   gabarito (só para --avaliar)
```

`data/json_to_submission.py` e `data/kaggle_metric.py` são os scripts oficiais do material.

### Comando que reproduz as saídas

```bash
uv sync --no-default-groups
uv run --no-default-groups python src/pipeline.py
```

Gera `saida/json/<documento_id>.json` (um por documento) e `saida/submission.csv`.
Com `--avaliar`, também pontua contra o gabarito usando a métrica oficial.

**Alternativa com Docker** (mesmas versões do `uv.lock`; os dados são montados, não copiados para a imagem):

```bash
docker build -t bracis-2026 .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/saida:/app/saida" bracis-2026
```

Argumentos após o nome da imagem vão para o pipeline (ex.: `... bracis-2026 --avaliar`).
No PowerShell, troque `$PWD` por `${PWD}`.

### Submeter no Kaggle

```bash
uv run python src/empacotar_submissao.py
```

Valida o `submission.csv` (cabeçalho, um `documento_id` por `.txt`, nenhuma célula
vazia, formato de cada citação) e cria `saida/submissao_<commit>/` com:

- `submission.csv` / `submission.zip`: o arquivo enviado no Kaggle
  (*Submit Prediction → File Upload*). Na descrição da submissão, anote o commit;
- `bundle_<commit>.zip`: o código exatamente nesse commit e um `comando.txt`,
  para a verificação de reprodutibilidade.

O script exige que as mudanças estejam commitadas, para o commit do nome ser o
que de fato gerou as saídas.

### Testes

```bash
uv sync --locked --no-default-groups --group dev
uv run --locked --no-default-groups --group dev pytest -q
```

Sem `data/desafio1_bracis.db`, os testes que dependem da base são pulados. A seção 6
do [SETUP.md](SETUP.md) descreve os gates de generalização (registro de regressão,
holdout sintético e base inteira), os dados necessários e o uso com WSL.

## Detalhes de implementação

### Configuração do pipeline

Os vocabulários e parâmetros editáveis ficam em `config/`: `tribunais.yaml`
(siglas e UFs), `classes_processuais.yaml` (marcadores e classes citadas),
`diplomas.yaml` (marcadores legais, apelidos e IDs da base), `regras_ocr.yaml`
(trocas de letras por dígitos) e `pipeline.yaml` (limiares e confiança).
Os arquivos usam o subconjunto JSON de YAML 1.2, lido sem dependências extras.
As mudanças são carregadas ao iniciar o processo; reinicie o pipeline após editar.
A ordem de `classes_citadas` e `apelidos_estatuto` é significativa, pois a primeira
correspondência prevalece. Reavalie com `python src/pipeline.py --avaliar` após
alterar os valores.

### Representação intermediária

Depois de extrair um span, o pipeline constrói uma `CitacaoNormalizada` com
família, identificadores e metadados extraídos, mantendo o trecho original e
seus offsets. O resolvedor consulta a base por esses campos e devolve um
`ResultadoResolucao` com classe, ID, candidatos, cobertura e motivo. Zero
registros compatíveis resulta em `inventada`; um resulta em `real`; vários ou
identificadores insuficientes resultam em `incompleta`. Para temas e leis
numeradas sem mapeamento, a convenção atual do desafio é preservada e a
cobertura fica marcada como `fora_base` no JSON de diagnóstico.
Lei, decreto ou resolução citados sem artigo resultam em `incompleta`, estejam
ou não no catálogo, porque a base registra dispositivos e não diplomas inteiros;
só um número de ato zerado continua `inventada`.

### Filtro contextual de metadados

Antes de filtrar candidatos, o pipeline identifica campos rotulados no início
do documento, o processo principal e a transição para o corpo. O filtro remove
spans contidos nesses campos, mesmo em cabeçalhos longos, e preserva referências
feitas no corpo. `motivo_distrator` informa a razão de cada exclusão; o limite
fixo de 400 caracteres não é mais usado.

## Estrutura do repositório

```text
config/             vocabulários e parâmetros do pipeline
data/               scripts oficiais e dados locais do desafio
docs/               descrição do desafio e documentação complementar
analise/             notebooks e registros usados nas análises
src/
├── pipeline.py      orquestra o processamento de ponta a ponta
├── steps/           etapas de extração, resolução e geração da saída
├── validacao/       gates de generalização e robustez
└── ml_citacoes/     experimentos opcionais de detecção e verificação
tests/               testes unitários, integração e regressão
saida/               JSONs, CSV e pacotes produzidos
```

O pipeline reproduzível descrito neste README usa o fluxo baseado em regras e
SQLite. Os módulos de `src/ml_citacoes/` são extensões experimentais opcionais e
não são necessários para gerar a submissão padrão.

## Documentação relacionada

- [Descrição completa do desafio](docs/descricao_desafio.md)
- [Instalação, execução e submissão](SETUP.md)
- [Estratégia de validação](src/validacao/README.md)
- [Geração dos documentos sintéticos](docs/GERACAO_DOCUMENTOS_SINTETICOS.md)
