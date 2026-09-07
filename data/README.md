# Dataset Description

### Arquivos
| Arquivo | O que é |
|---|---|
| `txt/` | Os 26 documentos de entrada (`.txt`, UTF-8) |
| `desafio1_bracis.db` | A base canônica SQLite (93 MB, 1.016 registros) |
| `goldenset.csv` | O gabarito da amostra de desenvolvimento (225 citações) |
| `json_to_submission.py` | Conversor: JSONs do contrato → `submission.csv` |
| `kaggle_metric.py` | O script de avaliação — o mesmo que roda neste leaderboard |

---
### A tarefa

**Encontrar as citações num documento judicial e classificar cada uma em três classes.**

A entrada é um `.txt` por documento. A saída é um JSON por documento, com uma entrada por citação: o span em codepoints Unicode (`inicio`, `fim` exclusivo), o trecho literal, o tipo (`lei` ou `jurisprudencia`) e a classe. Opcionalmente, a `confianca` (0–1) de que a classe — e o link, se `real` — está correta: ela alimenta o bônus de calibração de até 10%, e quem não a envia não ganha nem perde por isso.

> **Submissão no Kaggle:** sua solução produz os JSONs e o conversor `json_to_submission.py` (incluído aqui na aba Data) gera o `submission.csv` que você envia — uma linha por documento; documento sem citações leva `-` na célula, nunca vazia. A métrica que roda no servidor é a cópia exata de `kaggle_metric.py`, também nesta aba.

O alinhamento entre a predição e o gabarito é por sobreposição de spans com **IoU ≥ 0,5** — não é preciso acertar a borda exata, mas quem não entrega o span de uma citação não consegue classificá-la, e isso conta como erro de recall.
| `real` | `inventada` | `incompleta` |
|---|---|---|
| A citação resolve a um único **registro** da base canônica. Exige `id_canonico`: acertar o rótulo sem o doc_id correto não conta. | Identificadores suficientes para buscar, mas nenhum registro correspondente existe na cobertura congelada. | Informação insuficiente para formular a consulta, ou suficiente para buscar mas insuficiente para identificar um único registro. |

### Os documentos de entrada

**Vinte e seis peças jurídicas sintéticas, divididas em dois níveis de dificuldade — pesos 1× e 2× na nota.**

Esta é a **amostra de treino/desenvolvimento**, distribuída com gabarito para as equipes construírem e depurarem suas soluções — é sobre ela que o leaderboard roda na fase atual (referencial). O **conjunto de avaliação final** — cego — está **em construção** pela organização: terá o mesmo formato, os mesmos níveis e distribuição de classes equivalente, **sem gabarito distribuído**. Quando for ativado, os `.txt` dele serão publicados nesta aba, as saídas do seu sistema sobre eles passam a ser o que você submete, e é dele que saem o leaderboard (40% público) e o ranking final (60% privado).

São petições, pareceres, decisões monocráticas e memoriais fictícios que citam acórdãos reais do acervo. Cada documento tem uma matéria coerente — cível, penal, trabalhista, eleitoral ou militar — que amarra o cabeçalho, os tribunais citados e as normas invocadas: um agravo cível não cita apelação da Justiça Militar.

A nomenclatura dos arquivos já entrega o nível: `gen_n1_001` … `gen_n1_013` são os documentos do nível 1 e `gen_n2_001`  …  `gen_n2_013`, os do nível 2. O nome do arquivo em  `txt/`  (sem a extensão) é o `documento_id` usado no gabarito e na submissão.

|  | Nível 1 — Formato padrão | Nível 2 — Ruído e variação |
|---|---|---|
| **Peso na nota** | 1× | 2× |
| **Documentos** | 13 | 13 |
| **Citações** | 116 | 109 |
| **Tamanho médio** | 3.372 chars | 3.276 chars |
| **Classes** | real 52 · inventada 32 · incompleta 32 | real 44 · inventada 32 · incompleta 33 |
| **O que testa** | Reconhecer a citação e resolver o doc_id na base canônica. | Normalização robusta antes de verificar: casar variantes de superfície ao mesmo identificador. |

**A mesma classe, escrita de dois jeitos**

As citações do nível 2 apontam para registros igualmente válidos — o que muda é a superfície. Todas as amostras abaixo são real:

| Nível 1 | Nível 2 |
|---|---|
| `AREsp nº 1.996.496/RJ` | `AgRg no Rec. Esp. n. 1.522.200 (SC)` |
| `Recurso em Habeas Corpus nº 57.763/PR` | `Recurso em Habeas Corpus nº 93967 - SC` |
| `RSE nº 7000592-58.2025.7.00.0000/DF` | `Rec. Esp. No 1.880.529 ↵ - SP` |
| `Súmula Vinculante 10` | `Súmula 211 do STJ` |

O ruído do nível 2 combina variantes de abreviação (`REsp` / `R.Esp.` / `Recurso Especial`), formatação do número (`1.741.784` / `1741784` / `1.741. 784`), separador de UF (`/PR`, `- PR`, `(PR)`), confusões de OCR (`0↔O`, `1↔l`, `5↔S`, `m↔rn`) e quebras de linha no meio do identificador.

> **Garantia do ruído:** um dígito nunca é trocado por outro dígito. Isso mudaria a identidade da citação e transformaria uma `real` ruidosa numa `inventada` de fato. Todo ruído aplicado a uma citação real é recuperável por normalização — que é exatamente o que o nível 2 mede.

#### Distratores

Os cabeçalhos trazem números que _parecem_ citação e não são: número dos autos em formato CNJ, protocolo, inscrição na OAB, `fls. 234/567`, valor da causa. Nenhum está no gabarito — extraí-los conta como falso positivo.

Atenção à distinção no formato CNJ: o número dos autos _do próprio documento_, no cabeçalho, é distrator; a referência a _outro processo_ em formato CNJ, no corpo do texto, é citação (como o `RSE nº 7000592-58.2025.7.00.0000/DF` do quadro acima).

### A base canônica

**Um SQLite de 93 MB com os 1.016 registros que definem o universo do desafio. É contra ele que uma citação é real ou inventada.**

O banco é a **cobertura congelada**: um snapshot. Não se consulta base viva, e é isso que torna a avaliação reproduzível. Se um acórdão existe no mundo mas não está aqui, para efeito do desafio ele não existe — e, por construção, isso nunca prejudica ninguém: toda citação `real` dos documentos resolve dentro da cobertura. O caso-limite "registro real fora da cobertura" não ocorre neste dataset; se ocorresse, a citação seria excluída do scoring, não anotada como `inventada`.

| Coluna | Tipo | Descrição |
|---|---|---|
| `documento_id` | TEXT | Chave primária e nome do `.txt`. `doc_0201` para acórdãos; o próprio doc_id para súmulas e leis. |
| `id` | INTEGER | **O doc_id canônico do Jusbrasil.** É este número que vai em `resolucao.id_canonico`. |
| `tribunal` | TEXT | STF, STJ, TSE, TST, STM. Nulo para dispositivos de lei. |
| `ano`, `relator` | INT, TEXT | Só os acórdãos têm. |
| `natureza` | TEXT | acordao · sumula · dispositivo |
| `tipo` | TEXT | jurisprudencia · lei — o enum do contrato. |
| `texto` | TEXT | Inteiro teor. Idêntico a `txt/<documento_id>.txt`. |
| `texto_len` | INTEGER | Comprimento em caracteres. |

`natureza`  existe porque  `tipo`  sozinho não separa acórdão de súmula — os dois são  `jurisprudencia`.
| Natureza | Registros | O que são |
|---|---:|---|
| `acordao` | 998 | Acórdãos de STF, STJ, TSE, TST e STM. |
| `sumula` | 5 | Súmulas do STJ, STF e TST, incluindo vinculante. |
| `dispositivo` | 13 | Artigos de CPC, CC, CLT, CF/88, CPP, CPM, CDC, Código Eleitoral e LC 64/1990. |

Há ainda uma tabela virtual `documentos_fts` (FTS5, _external content_) indexando o texto integral dos 1.016 registros. Ela não duplica o conteúdo — aponta para `documentos` pelo `rowid`.

## O gabarito

**O `goldenset.csv` tem uma linha por citação esperada — 225 no total, nos 26 documentos.**

É o gabarito da amostra de desenvolvimento. O gabarito do conjunto final tem o mesmo formato, permanece com a organização e é usado exclusivamente na avaliação oficial.
| Coluna | Descrição |
|---|---|
| `nivel` | 1 ou 2 — define o peso na nota final. |
| `documento_id`, `citacao_id` | Identificam a citação. `gen_n1_004` + `g3` — o `n1`/`n2` no nome indica o nível do documento. |
| `inicio`, `fim` | Span em codepoints Unicode, fim exclusivo. Chave de junção com a predição. |
| `trecho` | Cópia literal de `texto[inicio:fim]`. Quebras de linha aparecem escapadas como `\n`. |
| `tipo` | `lei` ou `jurisprudencia`. |
| `classificacao` | A classe esperada. É o que se pontua. |
| `id_canonico` | O doc_id do registro que resolve a citação. Só nas `real`. |

## Como usar a base

**O fluxo tem três passos, e a maior parte da dificuldade está no primeiro.**

### 1 · Normalizar o identificador antes de consultar

O índice FTS usa o tokenizador `unicode61`, que quebra em qualquer caractere não alfanumérico. `1.741.784` vira três tokens — `1`, `741`, `784`. A consequência prática:
```sql
-- funciona: a busca por frase reproduz a sequência de tokens
SELECT d.documento_id, d.id, d.tribunal
  FROM documentos_fts JOIN documentos d ON d.rowid = documentos_fts.rowid
 WHERE documentos_fts MATCH '"1.741.784"';   -- 1 resultado

-- não funciona: sem os separadores, é um token só, que não existe no índice
 WHERE documentos_fts MATCH '1741784';        -- 0 resultados
```
Como o nível 2 entrega números sem pontuação com frequência, o pipeline precisa reconstruir a forma canônica: extrair só os dígitos e reagrupá-los de três em três a partir da direita. `"1 741 784"` como frase também casa, porque produz a mesma sequência de tokens.

Números no padrão CNJ — `0600216-46.2020.6.14.0022` — funcionam do mesmo jeito, com busca por frase.

### 2 · Separar o registro de quem apenas o cita

Esta é a armadilha que mais custa precisão. O FTS devolve _qualquer_ documento que mencione o número, e acórdãos citam uns aos outros o tempo todo. Uma busca por `"1.276.977"` devolve seis documentos do STF — e **nenhum deles é o RE 1.276.977**. Todos apenas o citam.

O sinal que separa os dois casos é a posição. No documento que _é_ o processo, o número está no cabeçalho, nas primeiras dezenas de caracteres. Nos que apenas citam, ele aparece no corpo:
| Documento | Posição | Contexto |
|---|---:|---|
| `doc_0201` | 20 | `RECURSO ESPECIAL Nº 1.741.784 - PR …` — é o processo |
| `doc_0001` | 48 | `AG.REG. NA RECLAMAÇÃO 76.532 RIO DE JANEIRO …` — é o processo |
| `doc_0149` | 490 | menciona o RE 1.276.977 na ementa — apenas cita |
| `doc_0001` | 957 | também cita o RE 1.276.977, mais adiante |

Uma abordagem mais robusta que o limiar de posição é fazer o parsing do cabeçalho: classe processual, número e UF vêm sempre nas primeiras linhas, num formato estável por tribunal. Vale construir esse índice uma vez, offline, e consultar por chave em vez de varrer o FTS a cada citação.

### 3 · Contar os candidatos e decidir a classe

Resolvida a consulta, a cardinalidade do resultado dá a classe quase diretamente. Cada citação `real` do desafio resolve para exatamente um registro da base — as duplicatas remanescentes do acervo não têm nenhuma citação apontando para elas.
| Candidatos encontrados | Classe | Saída |
|---|---|---|
| exatamente 1 | `real` | `resolucao.id_canonico` = a coluna `id` do registro |
| 0 | `inventada` | `resolucao: null` |
| 2 ou mais candidatos distintos, sem critério de desempate | `incompleta` | `resolucao: null` |

Uma citação sem identificadores suficientes para sequer formular a consulta — "jurisprudência pacífica desta Corte", "o dispositivo legal de regência" — é `incompleta` sem passar pelo banco. E uma referência do tipo "acórdão do STF de 2024, relatado pelo Ministro Fulano" é buscável, mas casa com dezenas de candidatos distintos: também `incompleta`, por falta de critério de desempate.
```sql
-- a consulta por relator+ano que caracteriza a incompleta "buscável"
SELECT COUNT(*) FROM documentos
 WHERE natureza = 'acordao'
   AND tribunal = 'STF' AND ano = 2024
   AND relator LIKE '%Dino%';   -- dezenas de candidatos
```

### Leis e súmulas resolvem por registro próprio

Buscar "Súmula 83 do STJ" no texto dos acórdãos devolve dezenas de documentos que a mencionam — nenhum deles é a súmula. Os 18 registros de natureza `sumula` e `dispositivo` existem para isso: são o alvo da resolução, não o texto que cita.
```pgsql
-- todos os registros que não são acórdão, com o texto normativo
SELECT documento_id, id, natureza, tipo, substr(texto, 1, 80)
  FROM documentos
 WHERE natureza IN ('sumula', 'dispositivo')
 ORDER BY natureza, id;
```

Como são poucos, o caminho prático é carregá-los em memória e casar por texto normalizado do dispositivo — `art. 373, I, do CPC` e `artigo 373, inciso I, do Código de Processo Civil` apontam para o mesmo `id`.

> **Não confunda as duas colunas de ID:** `documento_id` é a chave interna do acervo e o nome do arquivo. `id` é o doc\_id do Jusbrasil, e é ele que vai em `resolucao.id_canonico`. Entregar `doc_0201` onde se espera `2566535283` derruba a citação para erro, mesmo com a classe certa.

***

_Cobertura congelada em 1.016 registros. A métrica completa e o alinhamento por IoU estão implementados no `kaggle_metric.py`, incluído nesta aba (ver Overview → Evaluation)._
