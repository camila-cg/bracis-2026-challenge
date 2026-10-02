---
model_name: BERTimbau Large LoRA para detecção de citações jurídicas
base_model: neuralmind/bert-large-portuguese-cased
base_model_revision: aa302f6ea73b759f7df9cad58bd272127b67ec28
library_name: peft
pipeline_tag: token-classification
language:
- pt
license: mit
tags:
- base_model:adapter:neuralmind/bert-large-portuguese-cased
- lora
- transformers
- token-classification
- legal
---

# BERTimbau Large LoRA para detecção de citações jurídicas

Adaptador LoRA experimental para localizar, em textos jurídicos em português
brasileiro, menções a legislação (`LEI`) e jurisprudência (`JUR`). O modelo
produz rótulos BIO e foi criado como complemento opcional do pipeline do
desafio Jusbrasil × BRACIS 2026.

Este diretório contém o adaptador e a cabeça de classificação, não os pesos do
modelo-base. Para inferência, também é necessário obter o BERTimbau Large na
revisão exata informada abaixo.

## Detalhes do modelo

- **Desenvolvido e disponibilizado por:** contribuidores do repositório
  `bracis-2026-challenge`.
- **Tipo:** classificador de tokens BERT Large com adaptadores LoRA e cabeça
  linear para cinco rótulos BIO.
- **Idioma e domínio:** português brasileiro; documentos jurídicos com citações
  legislativas e jurisprudenciais.
- **Rótulos:** `O`, `B-LEI`, `I-LEI`, `B-JUR` e `I-JUR`.
- **Modelo-base:**
  [`neuralmind/bert-large-portuguese-cased`](https://huggingface.co/neuralmind/bert-large-portuguese-cased),
  revisão `aa302f6ea73b759f7df9cad58bd272127b67ec28`.
- **Licença do adaptador:** MIT; consulte [LICENSE](LICENSE).
- **Licença do modelo-base:** MIT; consulte [NOTICE](NOTICE) para atribuição e
  aviso de terceiros.
- **Código-fonte:**
  [`camila-cg/bracis-2026-challenge`](https://github.com/camila-cg/bracis-2026-challenge).

## Usos

### Uso pretendido

O adaptador identifica spans candidatos de citações jurídicas para posterior
normalização, consulta à base canônica e classificação pelo pipeline do
repositório. Ele pode ser usado para experimentação e reprodução do desafio.

O decoder Viterbi do repositório impõe transições BIO válidas. As pontuações são
derivadas do softmax local dos tokens e não representam probabilidades
calibradas de correção jurídica.

## Como usar

Na raiz do repositório, instale o grupo ML travado pelo `uv.lock` e execute o
pipeline. Garanta que os pesos do modelo-base, na revisão fixada, estejam
disponíveis no cache local:

```bash
uv sync --locked --no-default-groups --group ml
uv run --locked --no-default-groups --group ml --no-sync python src/pipeline.py \
  --modelo-citacoes artefatos/ml_large/bertimbau/final \
  --dispositivo-modelo auto \
  --limiar-lei 0.99 --limiar-jur 0.99
```

O caminho fornecido ao pipeline é a pasta `final`, que reúne este adaptador, o
tokenizer e o `manifesto.json`. Por padrão, o carregamento não baixa pesos da
internet implicitamente. `auto` prefere CUDA e recua para CPU. Use
`--dispositivo-modelo cuda` para exigir GPU ou `--dispositivo-modelo cpu` para
forçar CPU.

## Dados de treinamento

O snapshot foi retreinado com `fold=todos` em 52 documentos: 26 documentos do
conjunto original e 26 documentos sintéticos, conforme os IDs registrados em
`../manifesto.json`. O hash SHA-256 do arquivo de treinamento preparado é:

```text
793a3b6bcc6fcb78eff4d8aa95fd2843895461424c94525cbebe0e0798ff8952
```

Os arquivos brutos de treinamento não são redistribuídos neste diretório. Como
modelos podem memorizar trechos, a publicação dos pesos não elimina a necessidade
de revisar a origem e as permissões dos dados. 

## Procedimento de treinamento

- arquitetura-base BERT Large (24 camadas; aproximadamente 335 milhões de
  parâmetros);
- LoRA com rank 7, alpha 14 e dropout 0,075 nos módulos `query` e `value`;
- cabeça de classificação de tokens treinável;
- 20 épocas, taxa de aprendizado `1e-4` e weight decay `0,01`;
- batch 1, acumulação de gradiente 4 e batch efetivo 4;
- janelas de 512 tokens com stride 96;
- entropia cruzada ponderada e subamostragem de tokens `O`;
- aumento de ruído OCR habilitado para os documentos originais;
- seed `260926`, precisão FP32 e gradient checkpointing habilitado;
- duração registrada de 147,17 segundos; pico de VRAM alocada de 1.436,40 MiB
  e reservada de 1.508 MiB.



## Avaliação

Este snapshot final foi treinado com todos os documentos disponíveis e não tem
conjunto de validação nem métricas independentes registradas em seu manifesto.
Por isso, esta página não declara precisão, recall ou F1 para este artefato.

Resultados de experimentos anteriores, descritos na documentação do
repositório, não devem ser interpretados como estimativa de generalização deste
snapshot. Antes de uso real, avalie-o em documentos independentes, com análise
por tipo de citação, qualidade de OCR, tribunal, período e fonte documental.

## Vieses, riscos e limitações

- O conjunto de treinamento é pequeno e específico do desafio, o que aumenta o
  risco de memorização e baixa generalização.
- Não houve validação independente deste snapshot após o retreino em todos os
  documentos.
- Citações incomuns, abreviações não vistas, OCR severo e textos fora do domínio
  podem causar falsos positivos ou falsos negativos.
- O modelo detecta forma textual; ele não verifica vigência, autenticidade,
  aplicabilidade ou correção material da referência.
- O tokenizer e o modelo-base refletem vieses do corpus de pré-treinamento do
  BERTimbau, que não foram reavaliados neste projeto.



## Reprodutibilidade

Os parâmetros, IDs dos documentos, revisão do modelo-base, hash do dataset,
balanceamento e consumo de recursos estão em `../manifesto.json`. A configuração
de treino correspondente está em `config/ml_bertimbau_large.json` no
repositório.

## Citação do modelo-base

```bibtex
@inproceedings{souza2020bertimbau,
  author    = {Souza, Fábio and Nogueira, Rodrigo and Lotufo, Roberto},
  title     = {BERTimbau: Pretrained BERT Models for Brazilian Portuguese},
  booktitle = {Intelligent Systems},
  year      = {2020},
  publisher = {Springer International Publishing},
  pages     = {403--417}
}
```

## Contato

Abra uma issue no
[`camila-cg/bracis-2026-challenge`](https://github.com/camila-cg/bracis-2026-challenge/issues).

### Versão do framework

- PEFT 0.21.0
