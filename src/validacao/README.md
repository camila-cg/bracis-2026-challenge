# Validação e generalização

O dev set está saturado e já não estima sozinho o desempenho no conjunto final
cego. Esta pasta separa dois objetivos:

1. regressão e robustez das regras, usando dados conhecidos ou gerados; e
2. generalização do detector e da fusão, que exige um corpus independente e
   uma auditoria de vazamento antes de carregar o modelo.

O arquivo `datasets/dataset_citacoes_br_sintetico_2026_seed26092217.zip` **não é
holdout neural**: todos os seus 26 documentos constam entre os documentos de
treino do artefato final. `holdout_zip.py` continua útil como regressão das
regras e do contrato da métrica, mas seu resultado não prova generalização do
BERTimbau nem da solução híbrida.

## Scripts

### `manifesto_datasets.py` — proveniência e vazamento

Gera um manifesto determinístico com origem, papel (`treino`, `validacao` ou
`teste`), caminhos relativos, tamanhos, contagens e SHA-256 de cada documento.
O comando de auditoria recalcula esses valores e falha se os dados mudarem. Uma
sobreposição entre papéis por `documento_id` ou por conteúdo exato também é
fatal, inclusive quando o arquivo foi renomeado.

Exemplo de especificação:

```json
{
  "versao": 1,
  "datasets": [
    {
      "nome": "holdout-independente",
      "documentos": "data/holdout_independente/txt",
      "anotacoes": "data/holdout_independente/goldenset_offsets.csv",
      "origem": "descricao-da-origem",
      "papel": "teste"
    }
  ]
}
```

```bash
python src/validacao/manifesto_datasets.py gerar \
  --especificacao data/holdout_independente/datasets.json \
  --saida data/holdout_independente/manifesto.json
python src/validacao/manifesto_datasets.py auditar \
  --manifesto data/holdout_independente/manifesto.json
```

Crie da mesma forma um manifesto dos documentos que efetivamente alimentaram o
treino. O script não inventa um split e não transforma dados conhecidos em
holdout; origem e papel precisam ser declarados por quem fornece o corpus.

### `gate_hibrido.py` — ablação pareada com métrica oficial

Depois de receber um holdout realmente independente, executa `regras`, `modelo`
e `fusao` sobre os mesmos textos. O detector é chamado uma vez por documento e
as predições são reutilizadas nas duas ablações neurais. Exceção de inferência,
fallback, zero spans, vazamento ou regressão da fusão encerram o gate com erro.
Os três CSVs e `relatorio.json` ficam na pasta informada. O relatório registra
o commit, se a árvore estava limpa, os limiares e hashes dos manifestos,
goldenset, banco, modelo, configuração e submissões.

Os limiares são obrigatórios e não são escolhidos no holdout. Para reproduzir a
configuração oficial atual:

```bash
uv run --locked --no-default-groups --group ml --no-sync \
  python src/validacao/gate_hibrido.py \
  --txt data/holdout_independente/txt \
  --goldenset data/holdout_independente/goldenset_offsets.csv \
  --db data/desafio1_bracis.db \
  --modelo .hf-cache/bertimbau-large-citacoes-juridicas \
  --manifesto-avaliacao data/holdout_independente/manifesto.json \
  --manifesto-treino data/manifesto_treino.json \
  --limiar-lei 0.99 --limiar-jur 0.99 \
  --saida saida/gate-hibrido
```

O gate de não regressão exige que a fusão preserve o score geral e por nível e
não aumente `tau`. Igualdade aprova segurança, mas não autoriza afirmar que o
modelo melhora o sistema; para essa afirmação, o relatório precisa mostrar
ganho estrito sobre regras em um holdout independente.

### `holdout_zip.py` — regressão sintética conhecida

Adapta o ZIP versionado ao contrato oficial sem regravar o original. Use o modo
`raw` (`--manter-difusas --manter-duplicatas`) para preservar as 71 anotações ou
o padrão curado, cujas decisões metodológicas estão descritas no próprio
script. Em ambos os casos, trate o resultado como regressão, não como holdout
do detector.

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

## Resultados históricos (26/09/2026)

Estes números registram uma revisão anterior e não substituem a execução dos
gates no commit atual. Use os comandos acima para produzir evidência vigente.

| Script | Resultado |
|---|---|
| `estresse.py` | `original` 1.0992 · `ruido` 1.0865–1.0897 · `digito` 1.0983, τ = 0 |
| `teste_base_inteira.py` | 717 de 988 registros resolvidos (72,6%). Principais falhas: cabeçalhos do TSE lidos como classe (45 `inventada_errado`, 61 `incompleta`); `AIRR`/`Ag-AIRR`/`RRAg` do TST e `Embargos Infringentes e de Nulidade` do STM sem extração; números compartilhados por mais de um registro caindo em `incompleta`. |
