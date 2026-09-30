"""
Pipeline de ponta a ponta — Desafio 1, BRACIS 2026 / Jusbrasil.

Etapas, na ordem em que cada citação passa por elas (uma por módulo em src/steps/):

    Por documento (processar_documento):
      1. extrair               txt -> candidatos a citação (detectar_citacoes)
      2. filtrar_distratores   descarta R$, fls., OAB, autos do próprio processo
      3. rotear                decide o tipo: lei, súmula, tema, descritiva, acórdão
      4. normalizar            extrai atributos e corrige OCR em CitacaoNormalizada
      5. resolver              consulta por atributos -> ResultadoResolucao
      6. atribuir_confianca    confiança pelo caminho de decisão da etapa 5

    Para o conjunto (main):
      7. gravar_saida          um JSON por documento -> submission.csv
      +  avaliar (opcional)    pontua com data/kaggle_metric.py (--avaliar)

Uso (da raiz do repositório):

    python src/pipeline.py                       # data/txt -> saida/json + saida/submission.csv
    python src/pipeline.py --avaliar             # idem, e pontua com data/kaggle_metric.py
    python src/pipeline.py --txt <pasta> --saida <pasta>

Nota: Nada aqui olha o gabarito: --avaliar só lê o goldenset depois que as predições estão prontas.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eda_utils as u
import indice_utils as ind  # noqa: F401 — usado por src/validacao via p.ind
from steps.pipeline_utils import DADOS, RAIZ
from steps.extrair_candidatos import extrair                  # 1
from steps.filtrar_distratores import e_distrator, filtrar_distratores     # 2
from steps.rotear_citacao import rotear                       # 3
from steps.normalizar_citacao import normalizar_citacao        # 4
from steps.buscar_na_base import Resolvedor                    # 5
from steps.atribuir_confianca import atribuir_confianca       # 6
from steps.salvar_submissao import gravar_saida               # 7
from steps.avaliar_metrica import avaliar, imprimir_avaliacao


def ler_texto(arquivo: Path) -> str:
    bruto = arquivo.read_bytes()
    if b"\r\n" in bruto:
        raise ValueError(f"{arquivo.name} contém CRLF; os offsets seriam deslocados.")
    return bruto.decode("utf-8")


def processar_documento(
    documento_id: str,
    texto: str,
    resolvedor: Resolvedor,
    detector_bio=None,
    limiares_modelo: dict[str, float] | None = None,
    verificador_citacoes=None,
) -> dict:
    candidatos = extrair(texto)                                        # 1
    candidatos = filtrar_distratores(candidatos, texto)                # 2
    eventos_deteccao = []
    if detector_bio is not None:
        from ml_citacoes.fusao import fusionar

        limiares = limiares_modelo
        if limiares is None:
            limiares = ({"LEI": 0.0, "JUR": 0.0} if verificador_citacoes is not None
                        else {"LEI": 0.95, "JUR": 0.95})
        fusao = fusionar(
            candidatos,
            detector_bio.detectar(texto),
            texto,
            limiares=limiares,
            filtro_distrator=e_distrator,
            verificador=verificador_citacoes,
        )
        candidatos = list(fusao.candidatos)
        eventos_deteccao = list(fusao.eventos)
    citacoes = []
    for candidato in candidatos:
        rota = rotear(candidato)                                       # 3
        # O detector mantém o span enxuto para a métrica, mas um qualificador
        # contraditório pode estar imediatamente fora dele (``do TJSP``). A
        # normalização recebe apenas pequenas janelas de contexto; os offsets e
        # o trecho devolvidos continuam sendo os do candidato original.
        candidato_normalizacao = dict(candidato)
        candidato_normalizacao["_contexto_esquerda"] = texto[
            max(0, candidato["inicio"] - 48):candidato["inicio"]
        ]
        candidato_normalizacao["_contexto_direita"] = texto[
            candidato["fim"]:min(len(texto), candidato["fim"] + 48)
        ]
        citacao = normalizar_citacao(candidato_normalizacao, rota)      # 4
        resultado = resolvedor.resolver_citacao(citacao)               # 5
        confianca = atribuir_confianca(resultado.caminho, citacao.fonte)  # 6
        saida = {
            "inicio": candidato["inicio"],
            "fim": candidato["fim"],
            "trecho": candidato["trecho"],
            "tipo": candidato["tipo"],
            "classificacao": resultado.classe,
            "resolucao": {"id_canonico": str(resultado.id_canonico)} if resultado.id_canonico is not None else None,
            "confianca": confianca,
            "caminho": resultado.caminho,
            "cobertura": resultado.cobertura,
            "motivo": resultado.motivo,
            "fonte": citacao.fonte,
        }
        if "evidencia_deteccao" in candidato:
            saida["evidencia_deteccao"] = candidato["evidencia_deteccao"]
        citacoes.append(saida)
    documento = {"documento_id": documento_id, "citacoes": citacoes}
    if detector_bio is not None:
        documento["eventos_deteccao"] = eventos_deteccao
        documento["decodificador_bio"] = detector_bio.decodificador
    return documento


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--txt", type=Path, default=DADOS / "txt")
    parser.add_argument("--db", type=Path, default=DADOS / "desafio1_bracis.db")
    parser.add_argument("--saida", type=Path, default=RAIZ / "saida")
    parser.add_argument("--avaliar", action="store_true", help="pontua contra data/goldenset_offsets.csv")
    parser.add_argument("--goldenset", type=Path, default=DADOS / "goldenset_offsets.csv")
    parser.add_argument("--modelo-citacoes", type=Path, help="artefato LoRA local; omitido mantém regras idênticas")
    parser.add_argument("--decodificador-bio", choices=["viterbi", "argmax"], default="viterbi",
                        help="Viterbi padrão; argmax disponível para comparação histórica")
    parser.add_argument("--lote-janelas-bio", type=int, default=8)
    parser.add_argument("--limiar-lei", type=float)
    parser.add_argument("--limiar-jur", type=float)
    parser.add_argument("--verificador-llm", type=str,
                        help="pasta local ou ID do Llama 3.1 8B Instruct quantizado em 4 bits")
    parser.add_argument("--verificador-ollama", help="modelo local do Ollama, por exemplo llama3.1:8b")
    parser.add_argument("--endpoint-ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--revisao-verificador", help="revisão exata obrigatória para ID remoto")
    parser.add_argument("--permitir-download-verificador", action="store_true")
    parser.add_argument("--lote-verificador", type=int, default=8)
    args = parser.parse_args()

    arquivos = sorted(args.txt.glob("*.txt"))
    if not arquivos:
        sys.exit(f"nenhum .txt em {args.txt}")
    con = sqlite3.connect(u._uri_sqlite(args.db.resolve()) + "?immutable=1", uri=True)
    resolvedor = Resolvedor(con)
    detector_bio = None
    if args.modelo_citacoes is not None:
        from ml_citacoes.inferencia import DetectorBIO

        detector_bio = DetectorBIO.de_artefato(
            args.modelo_citacoes,
            decodificador=args.decodificador_bio,
            lote_janelas=args.lote_janelas_bio,
        )
    verificador_citacoes = None
    if args.verificador_llm is not None and args.verificador_ollama is not None:
        sys.exit("use somente um backend de verificação")
    if args.verificador_llm is not None or args.verificador_ollama is not None:
        if detector_bio is None:
            sys.exit("o verificador requer --modelo-citacoes")
        if args.verificador_ollama is not None:
            from ml_citacoes.verificador_llm import VerificadorOllama
            verificador_citacoes = VerificadorOllama(
                modelo=args.verificador_ollama, endpoint=args.endpoint_ollama,
            )
        else:
            from ml_citacoes.verificador_llm import VerificadorLlama
            verificador_citacoes = VerificadorLlama.de_modelo(
                args.verificador_llm,
                revisao=args.revisao_verificador,
                permitir_download=args.permitir_download_verificador,
                lote=args.lote_verificador,
            )
    if (args.limiar_lei is None) != (args.limiar_jur is None):
        sys.exit("informe os dois limiares ou omita ambos")
    limiares_modelo = None if args.limiar_lei is None else {
        "LEI": args.limiar_lei, "JUR": args.limiar_jur,
    }

    documentos = [
        processar_documento(
            a.stem,
            ler_texto(a),
            resolvedor,
            detector_bio,
            limiares_modelo,
            verificador_citacoes,
        )
        for a in arquivos
    ]  # 1–6
    submission_csv = gravar_saida(documentos, args.saida)                                   # 7

    if args.avaliar:
        imprimir_avaliacao(avaliar(submission_csv, args.goldenset))


if __name__ == "__main__":
    main()
