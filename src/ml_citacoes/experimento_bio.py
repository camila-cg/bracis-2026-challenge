"""Ablação pareada de decodificação, exclusivamente no fold 0 já treinado.

Não treina, não busca limiar, não calibra, não consulta o conjunto final.
Relatório detalhado fica local; resumo não contém texto, IDs ou offsets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .avaliar_validacao import _com_documento, _detalhar_erros
from .bio import bio_para_spans, combinar_logits_janelas
from .fusao import fusionar, iou
from .inferencia import DetectorBIO
from .metricas import avaliar_spans
from steps.extrair_candidatos import extrair
from steps.filtrar_distratores import e_distrator, filtrar_distratores


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validar_entrada(dados, execucao):
    manifesto = json.loads((execucao / 'manifesto.json').read_text(encoding='utf-8'))
    if str(manifesto['fold']) != '0':
        raise ValueError('este smoke test aceita somente fold 0')
    if _hash(dados / 'documentos.jsonl') != manifesto['dataset_sha256']:
        raise ValueError('dataset diverge do usado no treinamento')
    docs = [json.loads(l) for l in (dados / 'documentos.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    por_id = {d['documento_id']: d for d in docs}
    if len(por_id) != len(docs):
        raise ValueError('documento_id duplicado')
    treino, validacao = set(manifesto['documentos_treino']), set(manifesto['documentos_validacao'])
    if not treino or not validacao or treino & validacao:
        raise ValueError('split vazio ou com vazamento por ID')
    if not (treino | validacao) <= por_id.keys():
        raise ValueError('documentos ausentes')
    folds = json.loads((dados / 'folds.json').read_text(encoding='utf-8'))
    fold = folds['folds'][0]
    if set(fold['treino']) != treino or set(fold['validacao']) != validacao:
        raise ValueError('manifestos de execução e de folds divergem')
    hashes_treino = {hashlib.sha256(por_id[i]['texto'].encode()).hexdigest() for i in treino}
    for doc in docs:
        if doc['documento_id'] in validacao:
            if doc.get('origem') == 'sintetico':
                raise ValueError('sintético na validação')
            if hashlib.sha256(doc['texto'].encode()).hexdigest() in hashes_treino:
                raise ValueError('texto idêntico entre treino e validação; auditar a origem')
        anterior = 0
        for s in sorted(doc['spans'], key=lambda x: (x['inicio'], x['fim'])):
            if not 0 <= s['inicio'] < s['fim'] <= len(doc['texto']) or s['inicio'] < anterior:
                raise ValueError('offset inválido ou anotação sobreposta')
            if 'trecho' in s and doc['texto'][s['inicio']:s['fim']] != s['trecho']:
                raise ValueError('trecho diverge dos offsets')
            anterior = s['fim']
    return manifesto, {i: por_id[i] for i in sorted(validacao)}


def _tipo(s):
    return 'LEI' if s['tipo'].upper() == 'LEI' else 'JUR'


def _fragmentacao(verdade, previsto):
    # Diagnóstico por sobreposição, não matching da métrica principal.
    afetados = sum(sum(p['documento_id'] == g['documento_id'] and _tipo(p) == _tipo(g)
                        and iou(g['inicio'], g['fim'], p['inicio'], p['fim']) > 0
                        for p in previsto) > 1 for g in verdade)
    return {'entidades_com_multiplos_fragmentos': afetados, 'total_entidades': len(verdade),
            'taxa': afetados / len(verdade) if verdade else None}


def _metricas(verdade, previsto, docs):
    resultado = {'iou_05': avaliar_spans(verdade, previsto),
                 'exato': avaliar_spans(verdade, previsto, 1.0),
                 'fragmentacao': _fragmentacao(verdade, previsto)}
    resultado['por_nivel'] = {}
    for nivel in sorted({str(d.get('nivel', 'desconhecido')) for d in docs.values()}):
        ids = {i for i, d in docs.items() if str(d.get('nivel', 'desconhecido')) == nivel}
        resultado['por_nivel'][nivel] = avaliar_spans(
            [g for g in verdade if g['documento_id'] in ids],
            [p for p in previsto if p['documento_id'] in ids])
    resultado['ocr_proxy_nivel_2'] = resultado['por_nivel'].get('2')
    return resultado


def _faixas_score(verdade, previsto, docs):
    erros = _detalhar_erros(verdade, previsto, docs)
    fp = {(p['documento_id'], p['inicio'], p['fim'], p['tipo']) for p in erros['falsos_positivos']}
    faixas = []
    for baixo, alto in ((0, .5), (.5, .8), (.8, .9), (.9, .99), (.99, 1.000001)):
        itens = [p for p in previsto if baixo <= p['score'] < alto]
        corretos = sum((p['documento_id'], p['inicio'], p['fim'], _tipo(p)) not in fp for p in itens)
        faixas.append({'min_inclusivo': baixo, 'max_exclusivo': alto, 'n': len(itens),
                       'tp': corretos, 'precisao': corretos / len(itens) if itens else None,
                       'score_medio': sum(p['score'] for p in itens) / len(itens) if itens else None})
    return faixas


def executar(args, detector=None):
    manifesto, docs = validar_entrada(args.dados, args.execucao)
    if args.saida.exists():
        raise ValueError('saída já existe; use uma nova pasta para preservar a ablação')
    limiares = {'LEI': args.limiar_lei, 'JUR': args.limiar_jur}
    if any(not 0 <= x <= 1 for x in limiares.values()):
        raise ValueError('limiares precisam estar entre 0 e 1')
    if detector is None:
        detector = DetectorBIO.de_artefato(args.execucao, permitir_download=False)
    verdade, regras = [], []
    previstos = {modo: {'bruto': [], 'limiar': [], 'fusao': [], 'adicionados': []} for modo in ('argmax', 'viterbi')}
    diagnosticos = {modo: {'transicoes_invalidas': 0, 'documentos_invalidos': 0,
                         'b_consecutivos_mesmo_tipo': 0, 'tokens': 0, 'tokens_positivos': 0} for modo in previstos}
    mudancas_tokens = 0
    pred_argmax = {}
    for doc_id, doc in docs.items():
        verdade.extend(_com_documento(s, doc_id) for s in doc['spans'])
        regras_doc = filtrar_distratores(extrair(doc['texto']), doc['texto'])
        regras.extend(_com_documento(s, doc_id) for s in regras_doc)
        offsets, logits = detector.obter_logits(doc['texto'])
        labels_argmax = None
        for modo in previstos:
            off, labels, scores = combinar_logits_janelas(offsets, logits, decodificador=modo)
            if modo == 'argmax':
                labels_argmax = labels
            else:
                mudancas_tokens += sum(a != b for a, b in zip(labels_argmax, labels))
            spans, diag = bio_para_spans(labels, off, scores)
            if any(not 0 <= s.inicio < s.fim <= len(doc['texto']) for s in spans):
                raise ValueError('previsão fora dos limites do documento')
            ps = [{'inicio': s.inicio, 'fim': s.fim, 'tipo': s.tipo, 'score': s.score_bruto} for s in spans]
            if modo == 'argmax':
                pred_argmax[doc_id] = ps
            previstos[modo]['bruto'].extend(_com_documento(s, doc_id) for s in ps)
            previstos[modo]['limiar'].extend(_com_documento(s, doc_id) for s in ps if s['score'] >= limiares[s['tipo']])
            fusao = fusionar(regras_doc, spans, doc['texto'], limiares=limiares, filtro_distrator=e_distrator)
            previstos[modo]['fusao'].extend(_com_documento(s, doc_id) for s in fusao.candidatos)
            previstos[modo]['adicionados'].extend(_com_documento(s, doc_id) for s in fusao.candidatos
                                                 if s['evidencia_deteccao']['origem'] == 'modelo')
            d = diagnosticos[modo]
            d['transicoes_invalidas'] += diag.invalidas
            d['documentos_invalidos'] += int(diag.invalidas > 0)
            d['b_consecutivos_mesmo_tipo'] += diag.fragmentos
            d['tokens'] += len(labels)
            d['tokens_positivos'] += sum(x != 0 for x in labels)
    regras_m = _metricas(verdade, regras, docs)
    erros_regras = _detalhar_erros(verdade, regras, docs)
    resultado = {'protocolo': 'exp01_bio_fold0_pareado', 'fold': 0,
                 'avaliacao_independente': False, 'motivo': 'fold já usado para escolher checkpoint e inspecionar erros',
                 'n_documentos': len(docs), 'n_entidades': len(verdade), 'limiares_fixos': limiares,
                 'calibracao': 'nenhuma; mínimo do softmax local do rótulo escolhido',
                 'ocr_definicao': 'proxy: metadado nivel=2; não é anotação independente de OCR',
                 'dataset_sha256': manifesto['dataset_sha256'], 'tokens_alterados': mudancas_tokens,
                 'regras': regras_m, 'variantes': {}, 'estabilidade_entre_folds': {'n_folds': 1, 'desvio': None}}
    detalhes = {'regras': erros_regras, 'variantes': {}}
    for modo, grupos in previstos.items():
        d = diagnosticos[modo]
        d['taxa_documentos_invalidos'] = d['documentos_invalidos'] / len(docs)
        d['taxa_transicoes_invalidas_por_token'] = d['transicoes_invalidas'] / max(1, d['tokens'])
        d['taxa_invalidas_por_token_positivo'] = d['transicoes_invalidas'] / max(1, d['tokens_positivos'])
        r = {nome: _metricas(verdade, grupos[nome], docs) for nome in ('bruto', 'limiar', 'fusao')}
        # Regras têm precedência: adições só recebem crédito pelo gold ainda não coberto.
        adicoes = avaliar_spans(erros_regras['falsos_negativos'], grupos['adicionados'])
        r['exclusivos_adicionados'] = adicoes
        r['recall_adicional_sobre_regras'] = adicoes['geral']['tp'] / len(verdade) if verdade else None
        r['fp_realmente_acrescentados'] = adicoes['geral']['fp']
        r['diagnostico_bio'] = d
        r['faixas_score_exploratorias'] = _faixas_score(verdade, grupos['bruto'], docs)
        r['colapso'] = len(grupos['bruto']) == 0 and len(verdade) > 0
        resultado['variantes'][modo] = r
        detalhes['variantes'][modo] = {
            nome: _detalhar_erros(verdade, grupos[nome], docs) for nome in ('bruto', 'limiar', 'fusao')}
        detalhes['variantes'][modo]['adicionados'] = _detalhar_erros(erros_regras['falsos_negativos'], grupos['adicionados'], docs)
    resultado['estabilidade_entre_folds']['medias_f1'] = {
        modo: {g: resultado['variantes'][modo][g]['iou_05']['geral']['f1']
               for g in ('bruto', 'limiar', 'fusao')} for modo in previstos}
    resultado['regras_perfeitas_em_recall'] = regras_m['iou_05']['geral']['fn'] == 0
    resultado['delta_viterbi_menos_argmax'] = {
        grupo: {k: resultado['variantes']['viterbi'][grupo]['iou_05']['geral'][k] - resultado['variantes']['argmax'][grupo]['iou_05']['geral'][k]
                for k in ('tp', 'fp', 'fn', 'precisao', 'recall', 'f1')}
        for grupo in ('bruto', 'limiar', 'fusao')}
    anterior = args.execucao / 'predicoes_validacao.json'
    if anterior.exists():
        antigas = json.loads(anterior.read_text(encoding='utf-8'))
        def chaves(ps):
            return sorted((p['inicio'], p['fim'], _tipo(p)) for p in ps)
        resultado['reproducao_argmax'] = {
            'documentos_com_spans_diferentes': sum(chaves(pred_argmax[i]) != chaves(antigas.get(i, [])) for i in docs),
            'documentos_ausentes_artefato_antigo': len(set(docs) - antigas.keys()),
            'max_diferenca_score_em_spans_iguais': max((
                abs(p['score'] - q['score']) for i in docs for p in pred_argmax[i]
                for q in antigas.get(i, []) if (p['inicio'], p['fim'], _tipo(p)) ==
                (q['inicio'], q['fim'], _tipo(q))), default=0.0)}
    else:
        resultado['reproducao_argmax'] = None
    versoes = {}
    for pacote in ('torch', 'transformers', 'peft', 'numpy'):
        try:
            versoes[pacote] = importlib.metadata.version(pacote)
        except importlib.metadata.PackageNotFoundError:
            versoes[pacote] = None
    resultado['versoes'] = versoes
    resultado['codigo_sha256'] = {p.name: _hash(p) for p in Path(__file__).parent.glob('*.py')}
    raiz = Path(__file__).resolve().parents[2]
    resultado['regras_config_sha256'] = {
        str(p.relative_to(raiz)): _hash(p)
        for p in sorted([*(raiz / 'src' / 'steps').glob('*.py'),
                         *(raiz / 'config').glob('*.yaml'),
                         raiz / 'src' / 'detectar_citacoes.py',
                         raiz / 'src' / 'indice_utils.py', raiz / 'src' / 'modelos.py'])
    }
    resultado['manifesto_sha256'] = _hash(args.execucao / 'manifesto.json')
    resultado['pesos_sha256'] = {p.name: _hash(p) for p in sorted((args.execucao / 'adaptador').glob('*')) if p.is_file()}
    resultado['treino_registrado'] = {'nota': 'global_step é o maior estado RETIDO; pode não ser a última atualização do treino', 'balanceamento': manifesto.get('balanceamento'), 'estado_disponivel': False}
    estados = sorted((args.execucao / 'checkpoints').glob('checkpoint-*/trainer_state.json'))
    if estados:
        estado = max((json.loads(p.read_text()) for p in estados), key=lambda s: s.get('global_step', 0))
        resultado['treino_registrado'].update(estado_disponivel=True, global_step=estado.get('global_step'),
            best_metric=estado.get('best_metric'), epoch=estado.get('epoch'),
            best_checkpoint=Path(estado['best_model_checkpoint']).name if estado.get('best_model_checkpoint') else None)
    args.saida.mkdir(parents=True, exist_ok=False)
    (args.saida / 'resumo.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.saida / 'erros_locais.json').write_text(json.dumps(detalhes, ensure_ascii=False, indent=2), encoding='utf-8')
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, required=True)
    parser.add_argument('--execucao', type=Path, required=True)
    parser.add_argument('--saida', type=Path, required=True)
    parser.add_argument('--limiar-lei', type=float, default=.99)
    parser.add_argument('--limiar-jur', type=float, default=.99)
    args = parser.parse_args()
    executar(args)
    print(f'Resumo agregado: {args.saida / "resumo.json"}')
    print('Erros detalhados preservados somente em erros_locais.json.')


if __name__ == '__main__':
    main()
