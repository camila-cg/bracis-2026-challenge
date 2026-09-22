# Verificação de citações jurídicas em pareceres de IA

Encontre citações de jurisprudência e lei em documentos e classifique cada uma como real, inventada ou incompleta. Jusbrasil × BRACIS 2026.

## Overview

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

### Filtro contextual de metadados

Antes de filtrar candidatos, o pipeline identifica campos rotulados no início
do documento, o processo principal e a transição para o corpo. O filtro remove
spans contidos nesses campos, mesmo em cabeçalhos longos, e preserva referências
feitas no corpo. `motivo_distrator` informa a razão de cada exclusão; o limite
fixo de 400 caracteres não é mais usado.

LLMs generativos já são usados para redigir pareceres, petições e memorandos jurídicos. Um dos riscos mais documentados desses sistemas é a **alucinação de fontes**: o modelo cita jurisprudência ou dispositivos legais que não existem, ou apresenta citações tão vagas que não podem ser verificadas. Nos últimos anos, tribunais no Brasil e no exterior já sancionaram advogados por protocolar peças com citações inventadas geradas por IA.

Neste desafio, você assume o papel de quem precisa construir o verificador automático que falta nesse fluxo de trabalho: um sistema que lê um documento judicial, identifica todas as citações de jurisprudência e de lei, e decide — para cada uma — se ela é **real**, **inventada** ou **incompleta**.

### A tarefa

Seu sistema recebe documentos judiciais em texto (`.txt`) e deve, para cada documento, identificar todas as citações e classificar cada uma em três classes:

* **`real`** — a citação resolve a um único registro da base canônica do desafio: tem identificadores suficientes e a busca confirma sua existência. Exige entregar o `id_canonico` correto — acertar o rótulo sem o link não conta.
* **`inventada`** — identificadores suficientes para buscar, mas nenhum registro correspondente existe na cobertura congelada.
* **`incompleta`** — informação insuficiente para formular a consulta ("conforme jurisprudência pacífica do tribunal"), ou suficiente para buscar mas insuficiente para identificar um único registro.

A distinção entre `inventada` e `incompleta` importa em produção: a primeira é alucinação ativa (o LLM inventou um número de acórdão); a segunda é evasiva. Sistemas reais tratam cada caso de forma diferente — incompletas vão para revisão humana; inventadas devem ser bloqueadas.

O conjunto tem **dois níveis de dificuldade**: Nível 1 (formato padrão, peso 1×) e Nível 2 (ruído de OCR e variação de superfície, peso 2×). Os detalhes de dados, base canônica e gabarito estão na aba **Data**.

### Como participar

1. **Monte sua equipe**: participação individual ou em equipes de até 4 pessoas (mesclagem de equipes pelo próprio Kaggle), apenas estudantes, do Brasil — a elegibilidade é verificada pela organização (ver Rules).
2. **Receba os dados**: documentos de entrada (`txt/`), base canônica SQLite e gabarito da amostra de desenvolvimento estão na aba **Data**, junto com o conversor de submissão e o script da métrica.
3. **Desenvolva sua solução** — apenas modelos e ferramentas de pesos e código **abertos** (veja Rules). Qualquer dataset público pode ser usado no treino.
4. **Submeta**: sua solução produz um JSON por documento (contrato na aba Data); o conversor `json_to_submission.py` gera o `submission.csv` que você envia aqui. A métrica oficial roda no servidor e o leaderboard atualiza na hora.
5. **Publique seu código**: as soluções do topo do ranking passam por verificação de reprodutibilidade ao final — o repositório e o commit que produziram as saídas serão coletados junto aos finalistas.

### Organização

O desafio é uma realização do **Jusbrasil** — empresa de tecnologia que conecta milhões de brasileiros à informação jurídica — em parceria com o **BRACIS 2026** — 36ª Brazilian Conference on Intelligent Systems, promovida pela Sociedade Brasileira de Computação.

Dúvidas sobre o desafio, a especificação técnica ou a logística: **desafio-bracis@jusbrasil.com.br**.

## Avaliação

A avaliação tem **duas fases**:

1. **Fase de treino (agora)** — o material distribuído é a amostra de desenvolvimento, com gabarito aberto. O leaderboard roda sobre ela e é **referencial**: serve para validar seu pipeline de ponta a ponta, não define o resultado.
2. **Fase de avaliação final** — o conjunto final, cego, está em construção pela organização. Quando for ativado, o leaderboard **reinicia**: passa a usar **40% dele** (parte pública do leaderboard), e o ranking final é calculado sobre os **60% restantes**, mantidos em sigilo até o encerramento — otimizar demais para o leaderboard não garante nada no resultado final. Submissões da fase de treino não contam para o ranking final.

O alinhamento entre suas citações e o gabarito é por **sobreposição de spans com IoU ≥ 0,5** (matching 1-para-1 pelo maior IoU). Citação do gabarito sem par vira erro de recall; predição sem par vira falso positivo. Sobre os pares casados, a nota de cada nível é montada em três passos:

**Passo 1 — macro-F1 das 3 classes.** F1 por classe (`real`, `inventada`, `incompleta`), com média simples entre elas — assim a classe `inventada`, rara mas central ao desafio, pesa tanto quanto a `real`. Para a classe `real`, só conta acerto se o `id_canonico` entregue for o doc\_id esperado no gabarito.

**Passo 2 — penalidade do erro grave.** O pior erro é carimbar uma alucinação como verificada:

```py
τ = fração das citações `inventada` do gabarito preditas como `real`
s = macroF1 · (1 − 0,5 · τ)
```

Se todas as alucinações "vazarem" como real, a nota cai pela metade.

**Passo 3 — bônus de calibração (até 10%).** O campo `confianca` é opcional; se enviado, medimos o Brier score sobre os pares casados e aplicamos `score = s · (1 + 0,10 · (1 − brier))`. Confiança alta quando acerta e baixa quando erra aproxima o bônus de +10%. Quem não envia confiança não ganha nem perde.

**Combinação dos níveis:**
```py
score_final = (1 · score_Nível1 + 2 · score_Nível2) / 3
```
Uma submissão perfeita com `confianca = 1.0` pontua **1.1000**.

### Formato de submissão

Sua solução produz um JSON por documento (contrato schema 1.2, na aba Data); o conversor `json_to_submission.py` (incluído no material) gera o `submission.csv` — **uma linha por documento**:
```python
documento_id,citacoes
gen_n1_001,"469,498,incompleta,-,0.9|589,652,incompleta,-,0.8"
gen_n1_002,-
```
Cada citação é `inicio,fim,classe,id_canonico,confianca` (campos ausentes viram `-`), com citações separadas por `|`. Documento **sem** citações leva `-` na célula — nunca deixe a célula vazia, e todo `documento_id` do conjunto precisa estar presente.

**Reproduza localmente:** o script de avaliação (`kaggle_metric.py`, na aba Data) é o mesmo que roda aqui — você reproduz na sua máquina exatamente o score do leaderboard, nível a nível, incluindo o matching.

## Cronograma e Premiação

### Cronograma
| Marco | Data |
|---|---|
| Abertura das inscrições | 18/08/2026 |
| Envio dos dados por e-mail aos inscritos | 25/08/2026 |
| Webinar de tira-dúvidas com a organização | 28/08/2026 |
| Período de submissões com leaderboard ao vivo | 01/09 a 30/09/2026 |
| Fechamento das submissões | 30/09/2026, 23h59 (BRT) |
| Apresentação das melhores soluções no BRACIS 2026 (Cuiabá-MT) | 19 a 22/10/2026 |

Todas as datas em horário de Brasília (BRT).

### Premiação e reconhecimento

* **Apresentação técnica no BRACIS 2026**: as 5 melhores soluções apresentam na sessão de encerramento da conferência, em Cuiabá-MT.
* **Contribuição à comunidade**: o desafio libera um benchmark público de NLP jurídico em português após a conferência; as soluções vencedoras entram como referência.

## FAQ

**Preciso estar inscrito no BRACIS para participar?** Não. Apenas as equipes vencedoras precisam comparecer (ou enviar um representante) à sessão de encerramento da conferência.

**Posso usar APIs comerciais, como modelos de linguagem pagos?** Não. Este desafio aceita **apenas modelos e ferramentas de pesos e código abertos** — a organização precisa conseguir executar sua solução de ponta a ponta, sem chaves de API nem serviços pagos. Servir um modelo de pesos abertos via API paga durante o desenvolvimento é permitido, desde que o mesmo modelo e revisão sejam executáveis offline pela organização. Qualquer dataset **público** pode ser usado no treino.

**Como funciona o leaderboard? Meu score final pode mudar?** Em duas fases. Agora, na fase de treino, o leaderboard pontua contra a amostra de desenvolvimento (gabarito aberto) e é referencial — use-o para validar seu pipeline. Quando o conjunto de avaliação final for ativado, o leaderboard reinicia e passa a usar a parte pública (40%) dele; o ranking final vem dos 60% privados, mantidos em sigilo até o encerramento — a posição pública é um bom sinal, mas o resultado final pode mudar.

**Quantas submissões posso fazer?** Múltiplas, com teto diário por equipe (configurado nesta competição).

**O offset (`inicio`/`fim`) pontua?** Não por si só — ele é a chave de junção com o gabarito (IoU ≥ 0,5). Acima do limiar, a borda exata é irrelevante; abaixo, a citação esperada fica sem par (erro de recall) e a predição órfã vira falso positivo. Cuidado com a contagem: codepoints Unicode sobre o texto exatamente como distribuído.

**O campo `confianca` é obrigatório?** Não, mas é recomendado: confiança bem calibrada (Brier baixo) rende bônus de até 10% no score.

**Em que língua são os documentos?** Português brasileiro, com citações de jurisprudência e legislação brasileiras.

