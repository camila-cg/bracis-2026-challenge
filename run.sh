#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "uso: docker run <imagem> <banco.db> <pasta_txt> <arquivo_saida.csv>" >&2
    exit 2
fi

db_path=$1
txt_path=$2
output_path=$3
output_dir=$(dirname "$output_path")
work_path=${BRACIS_WORK_DIR:-"$output_dir/.bracis-work"}

mkdir -p "$work_path" "$output_dir"

exec /app/.venv/bin/python /app/src/pipeline.py \
    --db "$db_path" \
    --txt "$txt_path" \
    --saida "$work_path" \
    --logs "$work_path/logs" \
    --arquivo-saida "$output_path" \
    --modelo-citacoes /opt/models/bertimbau-large-citacoes-juridicas \
    --dispositivo-modelo "${BRACIS_DISPOSITIVO:-cuda}" \
    --falhar-em-erros
