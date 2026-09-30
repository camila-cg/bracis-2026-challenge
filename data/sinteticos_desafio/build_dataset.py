"""Gera e valida o lote sintético de desafio para detecção de citações.

Este script apenas materializa dados estáticos. Ele não executa treinamento nem
consulta predições, folds de validação ou o conjunto final.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


RAIZ = Path(__file__).resolve().parent
REPOSITORIO = RAIZ.parents[2]


DOCUMENTOS = [
    {
        "documento_id": "sint_desafio_001",
        "classe_documento": "misto",
        "texto": (
            "RELATÓRIO DE DIGITALIZAÇÃO\n"
            "Protocolo 884120 — fls. 37/41.\n"
            "A parte invoca o art. 3O, § 2º, da Lei n.º 13.7O9/\n"
            "2O18 e, no mesmo parágrafo, o AgInt no R.Esp. n.º l.8O4.\n"
            "55O—RS. A menção genérica à legislação de proteção de dados não acrescenta outra fonte."
        ),
        "citacoes": [
            {
                "trecho": "art. 3O, § 2º, da Lei n.º 13.7O9/\n2O18",
                "tipo": "LEI",
                "dificuldade_introduzida": ["ruído plausível de OCR", "citação dividida por quebra de linha"],
                "justificativa": "O artigo, o parágrafo e o número/ano da lei permanecem recuperáveis apesar de O↔0 e da quebra de linha.",
            },
            {
                "trecho": "AgInt no R.Esp. n.º l.8O4.\n55O—RS",
                "tipo": "JUR",
                "dificuldade_introduzida": ["ruído plausível de OCR", "abreviação jurídica rara", "citação dividida por quebra de linha", "pontuação incomum"],
                "justificativa": "A classe recursal, o número recuperável e a UF individualizam o processo citado.",
            },
        ],
        "distratores": [
            ("Protocolo 884120", "Número de protocolo administrativo, sem espécie jurídica nem fonte citada."),
            ("fls. 37/41", "Indicação de folhas dos próprios autos."),
            ("legislação de proteção de dados", "Menção genérica a matéria legislativa, sem diploma individualizado."),
        ],
    },
    {
        "documento_id": "sint_desafio_002",
        "classe_documento": "misto",
        "texto": (
            "MINUTA\n"
            "OAB/AM 18.442 | valor controvertido: R$ 7.350,00.\n"
            "A motivação foi confrontada com o art. 489, §1º, inc. IV — CPC/15; e com a QO no IAC 15/STJ. "
            "Registrou-se ainda 'n.º 15' à margem, sem classe, órgão ou espécie."
        ),
        "citacoes": [
            {
                "trecho": "art. 489, §1º, inc. IV — CPC/15",
                "tipo": "LEI",
                "dificuldade_introduzida": ["pontuação incomum", "abreviação jurídica rara"],
                "justificativa": "Dispositivo, parágrafo, inciso e código/edição identificam a norma.",
            },
            {
                "trecho": "QO no IAC 15/STJ",
                "tipo": "JUR",
                "dificuldade_introduzida": ["abreviação jurídica rara", "citação curta identificável", "múltiplas citações próximas"],
                "justificativa": "A questão de ordem, o incidente numerado e o tribunal formam uma referência própria.",
            },
        ],
        "distratores": [
            ("OAB/AM 18.442", "Inscrição profissional, não fonte normativa ou precedente."),
            ("R$ 7.350,00", "Valor monetário."),
            ("n.º 15", "Número isolado, sem classe, órgão ou espécie jurídica."),
        ],
    },
    {
        "documento_id": "sint_desafio_003",
        "classe_documento": "positivo",
        "texto": (
            "NOTA TÉCNICA\n"
            "Segundo o acórdão da Segunda Turma do STJ no Recurso Especial 2.014.778/SC, julgado em 16 de maio de 2023, rel. Min. Assusete Magalhães, a controvérsia foi delimitada. "
            "A síntese também remete ao Tema 1.076/STJ e à SV 37."
        ),
        "citacoes": [
            {
                "trecho": "acórdão da Segunda Turma do STJ no Recurso Especial 2.014.778/SC, julgado em 16 de maio de 2023, rel. Min. Assusete Magalhães",
                "tipo": "JUR",
                "dificuldade_introduzida": ["citação descritiva individualizável"],
                "justificativa": "Órgão julgador, tribunal, classe, número, UF, data e relatora identificam a decisão no próprio trecho.",
            },
            {
                "trecho": "Tema 1.076/STJ",
                "tipo": "JUR",
                "dificuldade_introduzida": ["citação curta identificável", "múltiplas citações próximas"],
                "justificativa": "Tipo de precedente, número e tribunal bastam para identificar o tema.",
            },
            {
                "trecho": "SV 37",
                "tipo": "JUR",
                "dificuldade_introduzida": ["abreviação jurídica rara", "citação curta identificável", "múltiplas citações próximas"],
                "justificativa": "A abreviação de súmula vinculante e seu número apontam para uma fonte específica.",
            },
        ],
        "distratores": [],
    },
    {
        "documento_id": "sint_desafio_004",
        "classe_documento": "misto",
        "texto": (
            "ATO ADMINISTRATIVO\n"
            "Conselho Nacional de Justiça — expediente 455/2022.\n"
            "A providência observa a Res. CNJ nº 455, de 27-4-2022, bem como CF, art. 93, IX. "
            "A folha 455 e o número 93, lançados no rodapé, são controles internos."
        ),
        "citacoes": [
            {
                "trecho": "Res. CNJ nº 455, de 27-4-2022",
                "tipo": "LEI",
                "dificuldade_introduzida": ["abreviação jurídica rara", "pontuação incomum"],
                "justificativa": "Espécie normativa, órgão, número e data individualizam a resolução.",
            },
            {
                "trecho": "CF, art. 93, IX",
                "tipo": "LEI",
                "dificuldade_introduzida": ["citação curta identificável", "múltiplas citações próximas"],
                "justificativa": "Diploma, artigo e inciso identificam diretamente o dispositivo constitucional.",
            },
        ],
        "distratores": [
            ("Conselho Nacional de Justiça", "Órgão citado isoladamente, sem ato normativo associado neste trecho."),
            ("expediente 455/2022", "Número de expediente administrativo."),
            ("folha 455", "Controle de folha, não citação."),
            ("número 93", "Número solto em contexto de controle interno."),
        ],
    },
    {
        "documento_id": "sint_desafio_005",
        "classe_documento": "negativo",
        "texto": (
            "CERTIDÃO\n"
            "Processo 5019087-64.2026.4.03.6100\n"
            "Interessado representado por Ana L., OAB/SP 412.903. Juntada às fls. 118/121; depósito de R$ 12.480,70. "
            "A unidade do Tribunal Regional recebeu o feito e informou que a jurisprudência dominante, a legislação aplicável e a orientação sumulada serão examinadas oportunamente. "
            "Controle: 2026, 64, 03."
        ),
        "citacoes": [],
        "distratores": [
            ("Processo 5019087-64.2026.4.03.6100", "Número dos próprios autos no cabeçalho, não referência a outro processo."),
            ("OAB/SP 412.903", "Inscrição profissional."),
            ("fls. 118/121", "Indicação de folhas."),
            ("R$ 12.480,70", "Valor monetário."),
            ("Tribunal Regional", "Tribunal mencionado sem decisão, processo ou precedente individualizado."),
            ("jurisprudência dominante", "Alusão genérica a jurisprudência."),
            ("legislação aplicável", "Alusão genérica a legislação."),
            ("orientação sumulada", "Menção vaga, sem tribunal ou número de súmula."),
            ("2026, 64, 03", "Números soltos de controle."),
        ],
    },
    {
        "documento_id": "sint_desafio_006",
        "classe_documento": "negativo",
        "texto": (
            "MEMÓRIA DE REUNIÃO\n"
            "Foram anotados: 'Tema em debate: 27'; 'súmula do tribunal'; 'artigo seguinte do código'; 'acórdão de 2023'; e 'Lei estadual pertinente'. "
            "O marcador nº 42 identifica a caixa física, enquanto STJ e STF nomeiam destinatários de cópia. "
            "Nenhuma dessas expressões individualiza uma fonte."
        ),
        "citacoes": [],
        "distratores": [
            ("Tema em debate: 27", "Tema discursivo numerado, sem indicação de precedente ou tribunal."),
            ("súmula do tribunal", "Faltam número e tribunal determinado."),
            ("artigo seguinte do código", "Referência dêitica sem artigo nem diploma identificados."),
            ("acórdão de 2023", "Ano isolado não individualiza uma decisão."),
            ("Lei estadual pertinente", "Menção genérica sem número, data ou ente federativo."),
            ("nº 42", "Número de caixa física."),
            ("STJ", "Tribunal isolado."),
            ("STF", "Tribunal isolado."),
        ],
    },
    {
        "documento_id": "sint_desafio_007",
        "classe_documento": "misto",
        "texto": (
            "RAZÕES COMPLEMENTARES\n"
            "Autos 0007421-19.2025.5.04.0021 — fl. 9.\n"
            "A nulidade é sustentada nos arts. 5º, LIV e LV, da\n"
            "Constituição Federal de 1988; HC 241.907/SP; Rcl. 71.204/MG; e Tema 339/STF. "
            "O valor 241.907, lançado também como estimativa contábil, não constitui nova citação."
        ),
        "citacoes": [
            {
                "trecho": "arts. 5º, LIV e LV, da\nConstituição Federal de 1988",
                "tipo": "LEI",
                "dificuldade_introduzida": ["citação dividida por quebra de linha"],
                "justificativa": "Os incisos, o artigo e o diploma constitucional permanecem explícitos através da quebra.",
            },
            {
                "trecho": "HC 241.907/SP",
                "tipo": "JUR",
                "dificuldade_introduzida": ["citação curta identificável", "múltiplas citações próximas"],
                "justificativa": "Classe, número e UF individualizam o processo citado.",
            },
            {
                "trecho": "Rcl. 71.204/MG",
                "tipo": "JUR",
                "dificuldade_introduzida": ["pontuação incomum", "citação curta identificável", "múltiplas citações próximas"],
                "justificativa": "Classe, número e UF individualizam a reclamação.",
            },
            {
                "trecho": "Tema 339/STF",
                "tipo": "JUR",
                "dificuldade_introduzida": ["citação curta identificável", "múltiplas citações próximas"],
                "justificativa": "Tipo, número e tribunal identificam o precedente temático.",
            },
        ],
        "distratores": [
            ("Autos 0007421-19.2025.5.04.0021", "Número dos próprios autos no cabeçalho."),
            ("fl. 9", "Indicação de folha."),
            ("valor 241.907", "Número em contexto monetário/contábil, sem classe processual."),
        ],
    },
    {
        "documento_id": "sint_desafio_008",
        "classe_documento": "misto",
        "texto": (
            "EXCERTO DE PARECER\n"
            "Registro OAB/CE 55.701, lote 9830.\n"
            "Foram cotejados o Cód. de Proc. Penal-Militar, art. 5º, inc. II e o Decr. 9.83O, de 1O/6/2O19. "
            "Em seguida, examinou-se o AI nº 1024567-38.2023.8.26.0000/TJSP, cuja cópia estava parcialmente ilegível. "
            "A simples palavra 'resolução' no fecho não foi tomada por referência."
        ),
        "citacoes": [
            {
                "trecho": "Cód. de Proc. Penal-Militar, art. 5º, inc. II",
                "tipo": "LEI",
                "dificuldade_introduzida": ["abreviação jurídica rara", "pontuação incomum"],
                "justificativa": "O nome abreviado do código, artigo e inciso identificam o dispositivo.",
            },
            {
                "trecho": "Decr. 9.83O, de 1O/6/2O19",
                "tipo": "LEI",
                "dificuldade_introduzida": ["ruído plausível de OCR", "abreviação jurídica rara", "número normativo parcialmente degradado"],
                "justificativa": "Espécie, número e data continuam recuperáveis com a normalização O↔0.",
            },
            {
                "trecho": "AI nº 1024567-38.2023.8.26.0000/TJSP",
                "tipo": "JUR",
                "dificuldade_introduzida": ["número processual em formato CNJ", "citação curta identificável"],
                "justificativa": "Classe, número processual completo e tribunal identificam diretamente a referência.",
            },
        ],
        "distratores": [
            ("OAB/CE 55.701", "Inscrição profissional."),
            ("lote 9830", "Número de lote administrativo."),
            ("resolução", "Espécie normativa mencionada sem órgão, número, data ou objeto individualizador."),
        ],
    },
]


def localizar_unico(texto: str, trecho: str, documento_id: str) -> tuple[int, int]:
    ocorrencias = [m.start() for m in re.finditer(re.escape(trecho), texto)]
    if len(ocorrencias) != 1:
        raise ValueError(f"{documento_id}: esperado trecho único, encontrado {len(ocorrencias)}: {trecho!r}")
    inicio = ocorrencias[0]
    return inicio, inicio + len(trecho)


def ids_preexistentes() -> set[str]:
    ids: set[str] = set()
    for caminho in REPOSITORIO.rglob("manifest.csv"):
        if RAIZ in caminho.parents:
            continue
        with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
            for linha in csv.DictReader(arquivo):
                if linha.get("documento_id"):
                    ids.add(linha["documento_id"])
    for caminho in REPOSITORIO.rglob("documentos.jsonl"):
        if RAIZ in caminho.parents:
            continue
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                ids.add(json.loads(linha)["documento_id"])
    return ids


def montar() -> tuple[list[dict], list[dict], list[dict]]:
    documentos: list[dict] = []
    anotacoes: list[dict] = []
    distratores: list[dict] = []
    for especificacao in DOCUMENTOS:
        documento_id = especificacao["documento_id"]
        texto = especificacao["texto"]
        spans = []
        for citacao in especificacao["citacoes"]:
            inicio, fim = localizar_unico(texto, citacao["trecho"], documento_id)
            span = {
                "inicio": inicio,
                "fim": fim,
                "trecho": citacao["trecho"],
                "tipo": citacao["tipo"],
                "dificuldade_introduzida": citacao["dificuldade_introduzida"],
                "justificativa": citacao["justificativa"],
            }
            spans.append(span)
            anotacoes.append({"documento_id": documento_id, **span})
        spans.sort(key=lambda item: (item["inicio"], item["fim"]))
        documentos.append({
            "documento_id": documento_id,
            "texto": texto,
            "spans": spans,
            "nivel": "2",
            "origem": "sintetico_desafio",
            "classe_documento": especificacao["classe_documento"],
        })
        for trecho, justificativa in especificacao["distratores"]:
            inicio, fim = localizar_unico(texto, trecho, documento_id)
            distratores.append({
                "documento_id": documento_id,
                "inicio": inicio,
                "fim": fim,
                "trecho": trecho,
                "justificativa_rejeicao": justificativa,
            })
    return documentos, anotacoes, distratores


def validar(documentos: list[dict], anotacoes: list[dict], distratores: list[dict]) -> dict:
    erros_offsets: list[str] = []
    erros_sobreposicao: list[str] = []
    ids = [doc["documento_id"] for doc in documentos]
    duplicados = sorted(item for item, total in Counter(ids).items() if total > 1)
    colisoes = sorted(set(ids) & ids_preexistentes())
    textos = {doc["documento_id"]: doc["texto"] for doc in documentos}

    for doc in documentos:
        anterior = 0
        for span in doc["spans"]:
            inicio, fim = span["inicio"], span["fim"]
            if inicio < 0 or fim > len(doc["texto"]) or inicio >= fim:
                erros_offsets.append(f"{doc['documento_id']}:{inicio}:{fim}:limites")
            elif doc["texto"][inicio:fim] != span["trecho"]:
                erros_offsets.append(f"{doc['documento_id']}:{inicio}:{fim}:trecho")
            if inicio < anterior:
                erros_sobreposicao.append(f"{doc['documento_id']}:{inicio}:{fim}")
            anterior = max(anterior, fim)
    for item in distratores:
        if textos[item["documento_id"]][item["inicio"]:item["fim"]] != item["trecho"]:
            erros_offsets.append(f"distrator:{item['documento_id']}:{item['inicio']}:{item['fim']}")

    dificuldades = Counter(
        dificuldade
        for item in anotacoes
        for dificuldade in item["dificuldade_introduzida"]
    )
    classes_documento = Counter(doc["classe_documento"] for doc in documentos)
    tipos = Counter(item["tipo"] for item in anotacoes)
    resultado = {
        "resumo": {
            "documentos": len(documentos),
            "citacoes_total": len(anotacoes),
            "citacoes_LEI": tipos["LEI"],
            "citacoes_JUR": tipos["JUR"],
            "documentos_positivos": classes_documento["positivo"],
            "documentos_negativos": classes_documento["negativo"],
            "documentos_mistos": classes_documento["misto"],
            "distratores": len(distratores),
            "categorias_dificuldade": dict(sorted(dificuldades.items())),
        },
        "verificacoes": {
            "offsets_unicode_fim_exclusivo": "OK" if not erros_offsets else "FALHOU",
            "igualdade_texto_inicio_fim_trecho": "OK" if not erros_offsets else "FALHOU",
            "sobreposicao_spans": "OK" if not erros_sobreposicao else "FALHOU",
            "ids_duplicados_internos": duplicados,
            "colisoes_com_ids_preexistentes": colisoes,
            "prefixo_ids": "OK" if all(item.startswith("sint_desafio_") for item in ids) else "FALHOU",
            "origem": "OK" if all(doc["origem"] == "sintetico_desafio" for doc in documentos) else "FALHOU",
            "erros_offsets": erros_offsets,
            "erros_sobreposicao": erros_sobreposicao,
        },
    }
    if erros_offsets or erros_sobreposicao or duplicados or colisoes:
        raise ValueError(json.dumps(resultado["verificacoes"], ensure_ascii=False))
    return resultado


def escrever_jsonl(caminho: Path, itens: list[dict]) -> None:
    caminho.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in itens),
        encoding="utf-8",
    )


def main() -> None:
    documentos, anotacoes, distratores = montar()
    resultado = validar(documentos, anotacoes, distratores)
    escrever_jsonl(RAIZ / "documentos.jsonl", documentos)
    escrever_jsonl(RAIZ / "anotacoes.jsonl", anotacoes)
    escrever_jsonl(RAIZ / "distratores.jsonl", distratores)
    (RAIZ / "resumo_validacao.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
