#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "uso: docker run <imagem> <banco.db> <pasta_txt> <arquivo_saida.csv>" >&2
    exit 2
fi

db_path=$1
txt_path=$2
output_path=$3
work_path=${BRACIS_WORK_DIR:-/tmp/bracis-output}

mkdir -p "$work_path" "$(dirname "$output_path")"

exec /app/.venv/bin/python /app/src/pipeline.py \
    --db "$db_path" \
    --txt "$txt_path" \
    --saida "$work_path" \
    --arquivo-saida "$output_path" \
    --modelo-citacoes /opt/models/bertimbau-large-citacoes-juridicas
