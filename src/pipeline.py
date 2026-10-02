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
import json
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import eda_utils as u
import indice_utils as ind  # noqa: F401 — usado por src/validacao via p.ind
from steps.pipeline_utils import DADOS, RAIZ
from steps.extrair_candidatos import extrair, gravar_debug_extracao  # 1
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


def _registrar_erro(
    erros: list[dict],
    documento_id: str,
    etapa: str,
    exc: Exception,
    acao: str,
    *,
    arquivo: Path | None = None,
    contexto: dict | None = None,
) -> None:
    """Acumula uma falha recuperável sem incluir o conteúdo do documento."""
    registro = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "documento_id": documento_id,
        "etapa": etapa,
        "tipo_erro": type(exc).__name__,
        "mensagem": str(exc),
        "acao": acao,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    if arquivo is not None:
        registro["arquivo"] = arquivo.name
    if contexto:
        registro["contexto"] = contexto
    erros.append(registro)


def _resumir_candidato(candidato) -> dict:
    """Expõe apenas metadados seguros do candidato no log de erro."""
    if not isinstance(candidato, dict):
        return {"tipo_objeto": type(candidato).__name__}
    resumo = {}
    for chave in ("inicio", "fim", "tipo", "fonte"):
        if chave not in candidato:
            continue
        valor = candidato.get(chave)
        resumo[chave] = (
            valor
            if isinstance(valor, (str, int, float, bool)) or valor is None
            else type(valor).__name__
        )
    return resumo


def gravar_log_erros(
    erros: list[dict], pasta: Path, momento: datetime | None = None
) -> Path | None:
    """Grava um JSONL por execução; ausência de erros não cria arquivo vazio."""
    if not erros:
        return None
    momento = momento or datetime.now()
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"erros_pipeline_{momento:%Y-%m-%d_%H-%M-%S_%f}.jsonl"
    conteudo = "".join(json.dumps(erro, ensure_ascii=False) + "\n" for erro in erros)
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def processar_documento(
    documento_id: str,
    texto: str,
    resolvedor: Resolvedor,
    detector_bio=None,
    limiares_modelo: dict[str, float] | None = None,
    verificador_citacoes=None,
    debug_extracao: list[dict] | None = None,
    erros_processamento: list[dict] | None = None,
) -> dict:
    candidatos = extrair(texto)                                        # 1
    if debug_extracao is not None:
        debug_extracao.append({"documento_id": documento_id, "texto": texto,
                               "candidatos": list(candidatos)})
    candidatos = filtrar_distratores(candidatos, texto)                # 2
    eventos_deteccao = []
    detector_executado = False
    if detector_bio is not None:
        try:
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
            detector_executado = True
        except Exception as exc:
            if erros_processamento is None:
                raise
            _registrar_erro(
                erros_processamento,
                documento_id,
                "deteccao_modelo",
                exc,
                "fallback_para_regras",
            )
    citacoes = []
    for candidato in candidatos:
        etapa = "roteamento"
        try:
            rota = rotear(candidato)                                   # 3
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
            etapa = "normalizacao"
            citacao = normalizar_citacao(candidato_normalizacao, rota)  # 4
            etapa = "resolucao"
            resultado = resolvedor.resolver_citacao(citacao)           # 5
            etapa = "confianca"
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
        except Exception as exc:
            if erros_processamento is None:
                raise
            _registrar_erro(
                erros_processamento,
                documento_id,
                etapa,
                exc,
                "citacao_descartada",
                contexto=_resumir_candidato(candidato),
            )
    documento = {"documento_id": documento_id, "citacoes": citacoes}
    if detector_executado:
        documento["eventos_deteccao"] = eventos_deteccao
        documento["decodificador_bio"] = detector_bio.decodificador
    return documento


def processar_arquivo(
    arquivo: Path,
    resolvedor: Resolvedor,
    *,
    detector_bio=None,
    limiares_modelo: dict[str, float] | None = None,
    verificador_citacoes=None,
    debug_extracao: list[dict] | None = None,
    erros_processamento: list[dict],
) -> dict:
    """Processa um arquivo sem permitir que sua falha interrompa o lote."""
    documento_id = arquivo.stem
    try:
        texto = ler_texto(arquivo)
    except Exception as exc:
        _registrar_erro(
            erros_processamento,
            documento_id,
            "leitura",
            exc,
            "documento_emitido_sem_citacoes",
            arquivo=arquivo,
        )
        return {"documento_id": documento_id, "citacoes": []}

    try:
        return processar_documento(
            documento_id,
            texto,
            resolvedor,
            detector_bio,
            limiares_modelo,
            verificador_citacoes,
            debug_extracao,
            erros_processamento,
        )
    except Exception as exc:
        _registrar_erro(
            erros_processamento,
            documento_id,
            "processamento_documento",
            exc,
            "documento_emitido_sem_citacoes",
            arquivo=arquivo,
        )
        return {"documento_id": documento_id, "citacoes": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--txt", type=Path, default=DADOS / "txt")
    parser.add_argument("--db", type=Path, default=DADOS / "desafio1_bracis.db")
    parser.add_argument("--saida", type=Path, default=RAIZ / "saida")
    parser.add_argument("--logs", type=Path, default=RAIZ / "logs")
    parser.add_argument(
        "--arquivo-saida",
        type=Path,
        help="caminho exato do CSV final; o padrão é <saida>/submission.csv",
    )
    parser.add_argument("--avaliar", action="store_true", help="pontua contra data/goldenset_offsets.csv")
    parser.add_argument("--goldenset", type=Path, default=DADOS / "goldenset_offsets.csv")
    parser.add_argument(
        "--modelo-citacoes",
        type=Path,
        help="artefato local LoRA ou modelo mesclado; omitido mantém somente as regras",
    )
    parser.add_argument("--decodificador-bio", choices=["viterbi", "argmax"], default="viterbi",
                        help="Viterbi padrão; argmax disponível como alternativa")
    parser.add_argument("--lote-janelas-bio", type=int, default=8)
    parser.add_argument("--limiar-lei", type=float)
    parser.add_argument("--limiar-jur", type=float)
    parser.add_argument("--verificador-llm", type=str,
                        help="pasta local ou ID do Llama 3.1 8B Instruct quantizado em 4 bits")
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
    if args.verificador_llm is not None:
        if detector_bio is None:
            sys.exit("o verificador requer --modelo-citacoes")
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

    debug_extracao: list[dict] = []
    erros_processamento: list[dict] = []
    documentos = [
        processar_arquivo(
            a,
            resolvedor,
            detector_bio=detector_bio,
            limiares_modelo=limiares_modelo,
            verificador_citacoes=verificador_citacoes,
            debug_extracao=debug_extracao,
            erros_processamento=erros_processamento,
        )
        for a in arquivos
    ]  # 1–6
    log_erros = gravar_log_erros(erros_processamento, args.logs)
    if log_erros is not None:
        documentos_com_erro = len({erro["documento_id"] for erro in erros_processamento})
        print(
            f"atenção: {len(erros_processamento)} erro(s) recuperável(is) em "
            f"{documentos_com_erro} documento(s) -> {log_erros}"
        )
    print(f"debug da extração: {gravar_debug_extracao(debug_extracao, RAIZ / 'outputs')}")
    submission_csv = gravar_saida(
        documentos, args.saida, arquivo_csv=args.arquivo_saida,
    )                                                                                       # 7

    if args.avaliar:
        imprimir_avaliacao(avaliar(submission_csv, args.goldenset))


if __name__ == "__main__":
    main()
