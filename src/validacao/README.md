# Validação fora do dev set

O pipeline chega a 1.0992 (de 1.1000) no dev set, que por isso já não indica
como ele vai se sair no conjunto final cego. Os scripts desta pasta medem a
generalização de outras formas, sem ajustar nada ao gabarito.

Nenhum deles altera arquivos do repositório: só leem `data/` e imprimem no
terminal.

## Scripts

### `estresse.py` — robustez a ruído

Pega os 26 documentos, altera as citações do gabarito e mede o resultado
com a métrica oficial (`data/kaggle_metric.py`). Recalcula os offsets do
gabarito depois de cada alteração.

| Cenário | O que faz | Classe esperada |
|---|---|---|
| `original` | Texto como distribuído. Deve dar o mesmo score do `pipeline.py --avaliar`. | a do gabarito |
| `ruido` | Aplica às reais de acórdão o ruído do nível 2: abreviação (`REsp` ↔ `Recurso Especial`), formato do número (`1.741.784` / `1741784` / `1.741. 784`), separador de UF (`/PR`, `- PR`, `(PR)`), OCR (`0→O`, `1→l`, `5→S`) e quebra de linha. | continua `real` |
| `digito` | Troca um dígito do identificador das reais de acórdão. | passa a `inventada` |

Pergunta que responde: **a normalização e a resolução aguentam variações que
não estão no dev set?** E, no cenário `digito`, alguma alucinação vaza como
`real` (τ > 0)?

### `teste_base_inteira.py` — cobertura da base canônica

Para cada acórdão que o índice de cabeçalhos consegue ler, escreve uma frase
citando esse registro (`... no {classe} nº {número}/{UF}, ...`) e verifica se o
pipeline extrai o span e resolve para o `id` certo. As falhas são agrupadas
por tribunal, desfecho e classe processual:

- `nao_extraiu`: o extrator não achou citação (em geral, classe fora do vocabulário);
- `incompleta`: o número bate com mais de um registro e o desempate não decidiu;
- `real_errado` / `inventada_errado`: resolveu para outro registro ou para nenhum.

Pergunta que responde: **se o conjunto final citar registros que o dev set
não cita, quantos o pipeline acerta?**

É uma aproximação. A classe entra por extenso, como está no cabeçalho, e não
abreviada como apareceria num documento real. Os cabeçalhos do TSE que o
parser lê mal geram frases estranhas. Os registros sem parser ficam fora do
teste; o script só informa quantos são.

### `lookup_isolado.py` — índice e resolvedor, sem extração

Para cada acórdão do índice, monta direto uma citação com a chave que o próprio
índice guarda (tribunal, classe, número, UF) e pede ao resolvedor o registro
correspondente, sem passar pela extração de spans nem pela normalização de
texto. Desfechos:

- `ok`: `real` com o id esperado;
- `ambiguo_com_esperado`: `incompleta` com o id esperado entre os candidatos
  (resposta correta quando há duplicata no acervo);
- `ambiguo_sem_esperado`, `real_errado` e `inventada`: falhas do índice ou da
  regra de resolução.

Pergunta que responde: **o índice e a regra de resolução entregam o registro
certo quando a citação chega perfeita?** Comparado ao `teste_base_inteira.py`,
mostra em que etapa um registro se perde: se aqui acerta e lá não, a falha está
antes do índice (extração ou normalização).

## Como executar

Da raiz do repositório. Os scripts precisam de `data/desafio1_bracis.db`
e `data/txt/`, e o `estresse.py` também de `data/goldenset_offsets.csv` e
`data/kaggle_metric.py`.

`estresse.py` usa `pandas` e `numpy` (dependências da métrica oficial):

```bash
pip install pandas numpy
```

```bash
python src/validacao/estresse.py
```

Opção: `--seeds N` define quantas variações aleatórias gerar por cenário
perturbado (padrão 3).

`teste_base_inteira.py` só usa a biblioteca padrão:

```bash
python src/validacao/teste_base_inteira.py
```

Opções: `--top N` define quantos grupos de falha listar (padrão 25) e
`--exemplos N` quantas frases de exemplo mostrar (padrão 8).

`lookup_isolado.py` também só usa a biblioteca padrão:

```bash
python src/validacao/lookup_isolado.py
```

Opção: `--exemplos N` define quantos registros de exemplo das falhas mostrar
(padrão 8).

Sem instalar nada no ambiente, com `uv`:

```bash
uv run --no-project --with pandas --with numpy python src/validacao/estresse.py
```

## Resultados de referência (26/09/2026)

| Script | Resultado |
|---|---|
| `estresse.py` | `original` 1.0992 · `ruido` 1.0865–1.0897 · `digito` 1.0983, τ = 0 |
| `teste_base_inteira.py` | 717 de 988 registros resolvidos (72,6%). Principais falhas: cabeçalhos do TSE lidos como classe (45 `inventada_errado`, 61 `incompleta`); `AIRR`/`Ag-AIRR`/`RRAg` do TST e `Embargos Infringentes e de Nulidade` do STM sem extração; números compartilhados por mais de um registro caindo em `incompleta`. |
