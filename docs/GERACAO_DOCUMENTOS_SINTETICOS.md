# Geração de documentos sintéticos

O gerador cria documentos jurídicos sintéticos auditáveis a partir dos 26 arquivos de desenvolvimento. Ele preserva a estrutura das peças, transforma as citações anotadas e acrescenta casos controlados obtidos da base canônica.

## Modelo usado

O corpus usa o gerador **`procedural-regras-v1`**, uma implementação procedural
determinística baseada em regras. A mesma entrada, versão do código e semente
produzem a mesma saída.

Essa abordagem permite calcular os spans durante a montagem, conhecer a origem
de cada rótulo e auditar todas as transformações.

## Fontes e rótulos

- Os 26 `.txt` fornecem a estrutura dos documentos.
- `goldenset_offsets.csv` fornece as citações já anotadas e suas classes.
- `desafio1_bracis.db` fornece referências reais adicionais. Apenas chaves canônicas que resolvem para um único registro são usadas.
- Citações inventadas são produzidas pela alteração controlada de um dígito e verificadas contra os números conhecidos no índice.
- Citações incompletas são expressões sem identificadores suficientes para consulta.
- Documentos negativos substituem todas as citações anotadas por texto neutro.

## Prevenção de vazamento

O split acontece por documento-base antes da geração. Todas as versões de um mesmo original permanecem em `train`, `validation` ou `test`. Isso impede que uma variação de um documento apareça no treino enquanto outra versão quase idêntica aparece no teste.

Com 26 originais, a divisão padrão é aproximadamente 70%/15%/15%. A semente padrão é `20260922`.

## Perfis gerados

As variantes percorrem seis perfis:

1. transformações de superfície nas citações existentes;
2. inclusão de uma referência real unívoca da base canônica;
3. inclusão de uma referência inventada;
4. inclusão de uma referência incompleta;
5. inclusão de distratores de cabeçalho;
6. documento sem citações.

As transformações de superfície incluem marcador de número, quebra de linha, UF com hífen, ruído OCR reversível, Unicode NFD, remoção de separador de milhares e espaçamento duplicado.

## Execução

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe src\gerar_documentos_sinteticos.py
```

Para gerar outro lote:

```powershell
.\.venv\Scripts\python.exe src\gerar_documentos_sinteticos.py `
  --saida data\sinteticos_seed_42 `
  --variacoes 10 `
  --seed 42
```

Por segurança, o gerador recusa uma pasta de saída que já contenha arquivos. Use outra pasta para uma nova execução.

## Saída

```text
data/sinteticos/
├── txt/
│   ├── train/
│   ├── validation/
│   └── test/
├── goldenset_offsets.csv
├── manifest.jsonl
└── relatorio_validacao.json
```

O manifesto registra documento-base, split, semente, perfil e cada citação individual, incluindo origem, transformação, classe, ID canônico e chave do catálogo quando aplicável. O relatório resume a distribuição e o resultado das validações.

## Validações automáticas

Antes de concluir, o gerador verifica:

- limites e ausência de sobreposição dos spans;
- igualdade exata entre `texto[inicio:fim]` e o trecho anotado;
- presença do `id_canonico` de toda citação real na base;
- ausência de `id_canonico` nas classes inventada e incompleta;
- separação dos documentos derivados pelo documento-base.

O corpus sintético amplia a diversidade conhecida, mas não substitui um conjunto final formado por documentos independentes. O teste mais confiável continua sendo composto por peças de outras origens que não tenham participado da definição das regras.
