"""
Etapas do pipeline, uma por módulo, na ordem em que cada citação passa:

    1. extrair_candidatos    txt -> candidatos a citação (detectar_citacoes)
    2. filtrar_distratores   descarta R$, fls., OAB, cabeçalho do próprio processo
    3. rotear_citacao        decide o tipo: lei, súmula, tema, descritiva, acórdão
    4. normalizar_numeros    corrige OCR e reescreve números na forma do índice
    5. buscar_na_base        busca na base canônica -> classe + id_canonico
    6. atribuir_confianca    confiança pelo caminho de decisão da etapa 5
    7. salvar_submissao      um JSON por documento -> submission.csv
    +  avaliar_metrica       pontua com data/kaggle_metric.py (opcional)

    pipeline_utils           caminhos, OCR e blocos numéricos usados por várias etapas

A orquestração fica em src/pipeline.py.
"""
