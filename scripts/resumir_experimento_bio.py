"""Extrai apenas agregados do smoke test; não lê documentos nem erros locais."""
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
# O resumo já exclui IDs, offsets e textos. Preserva tipos, níveis e scores.
print(json.dumps(r, ensure_ascii=False, indent=2))
