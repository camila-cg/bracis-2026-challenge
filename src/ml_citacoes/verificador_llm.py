"""Classifica candidatos do LoRA por tipo jurídico usando o próximo token da LLM."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from .contratos import ResultadoVerificacao


ROTULOS = {
    "A": "CITACAO_LEGISLACAO",
    "B": "CITACAO_JURISPRUDENCIA",
    "C": "INCERTO",
    "D": "NAO_CITACAO",
}
PROMPT_VERSAO = "zero_shot_v3_quatro_classes"

PROMPT_SISTEMA = """Você é um especialista em direito brasileiro e deve classificar o trecho fornecido, sem tentar completar ou corrigir seu conteúdo.

Responda com exatamente uma letra, dentre as opções:

A = citação de legislação;
B = citação de jurisprudência;
C = incerto;
D = não é uma citação jurídica identificável.

Uma citação identificável aponta, no próprio trecho, para uma fonte jurídica específica, como norma, processo, precedente, súmula, tema ou decisão. Não é necessário confirmar que a fonte realmente existe.

Classifique como A quando o trecho identificar uma fonte normativa específica:

- espécie normativa e número identificador, acrescentando órgão, ente ou ano quando necessários para eliminar ambiguidade;
- nome oficial, nome consagrado ou sigla inequívoca da norma desde que acompanhada de número normativo, dispositivo ou outra referência concreta;
- artigo, parágrafo, inciso ou outro dispositivo acompanhado da identificação do diploma ao qual pertence;
- referência normativa reconhecível apesar de ruído de OCR, quebra de linha ou pontuação corrompida, desde que ainda seja possível reconhecer o diploma, a espécie normativa ou seu identificador.

Não classifique como A números isolados, dispositivos sem diploma ou menções genéricas a lei, código, norma, regulamento ou legislação.

Classifique como B quando o trecho identificar uma fonte jurisprudencial concreta por pelo menos uma destas formas:

- por classe, número, tema ou súmula numerada; ou
- decisão descritiva sem número processual que contenha obrigatoriamente:
   1. tribunal ou órgão judicial específico;
   2. ano ou data;
   3. relator identificado.
- referência jurisprudencial reconhecível apesar de ruído de OCR, quebra de linha ou pontuação corrompida, desde que ainda seja possível reconhecer a classe, o precedente ou parte suficiente do identificador.
Ruído de OCR, quebra de linha ou pontuação corrompida podem ser tolerados, mas
não eliminam os requisitos acima.

A ausência de número não torna automaticamente uma citação inválida. Entretanto, palavras como processo, recurso, acórdão, decisão, tema, precedente ou súmula não são suficientes quando aparecem sem elementos que apontem para uma fonte concreta.

Classifique como D números soltos, fragmentos, nomes isolados de tribunais ou órgãos e menções genéricas a jurisprudência, legislação, matéria ou orientação.
Classifique como D quando o trecho não fala de nenhuma fonte específica, apenas remete, de forma difusa, à matéria ou à orientação de um tribunal, mas sem permitir identificá-lo."""

PROMPT_SHA256 = hashlib.sha256(PROMPT_SISTEMA.encode("utf-8")).hexdigest()


def _softmax_restrito(valores: Sequence[float]) -> list[float]:
    if len(valores) != len(ROTULOS) or any(not math.isfinite(float(v)) for v in valores):
        raise ValueError(f"são necessários {len(ROTULOS)} logits finitos")
    maior = max(float(v) for v in valores)
    exponenciais = [math.exp(float(v) - maior) for v in valores]
    total = sum(exponenciais)
    return [valor / total for valor in exponenciais]


def _resultado_de_logits(logits: Sequence[float], *, modelo: str) -> ResultadoVerificacao:
    probabilidades = _softmax_restrito(logits)
    letras = tuple(ROTULOS)
    maior = max(probabilidades)
    empatados = [i for i, valor in enumerate(probabilidades) if math.isclose(valor, maior)]
    indice = letras.index("C") if len(empatados) > 1 else empatados[0]
    probs_nomeadas = {ROTULOS[letra]: probabilidades[i] for i, letra in enumerate(letras)}
    return ResultadoVerificacao(
        decisao=ROTULOS[letras[indice]],
        confianca=probabilidades[indice],
        probabilidades=probs_nomeadas,
        modelo=modelo,
        prompt_sha256=PROMPT_SHA256,
    )


class VerificadorLlama:
    """Llama 3.1 Instruct congelado e quantizado em 4 bits."""

    def __init__(self, tokenizer, modelo, *, nome_modelo: str, lote: int = 8):
        if lote < 1:
            raise ValueError("lote deve ser positivo")
        self.tokenizer = tokenizer
        self.modelo = modelo
        self.nome_modelo = nome_modelo
        self.lote = lote
        self.ids_rotulos = tuple(self._id_unitario(letra) for letra in ROTULOS)

    @classmethod
    def de_modelo(
        cls,
        caminho: str | Path,
        *,
        revisao: str | None = None,
        permitir_download: bool = False,
        lote: int = 8,
    ) -> "VerificadorLlama":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "verificação LLM requer torch, transformers, accelerate e bitsandbytes"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("o verificador Llama 4-bit requer uma GPU CUDA")

        referencia = str(caminho)
        local = Path(referencia).exists()
        if not local and not revisao:
            raise ValueError("informe uma revisão exata ao usar um modelo remoto")
        local_files_only = local or not permitir_download
        tokenizer = AutoTokenizer.from_pretrained(
            referencia,
            revision=None if local else revisao,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantizacao = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        modelo = AutoModelForCausalLM.from_pretrained(
            referencia,
            revision=None if local else revisao,
            local_files_only=local_files_only,
            quantization_config=quantizacao,
            device_map="auto",
            torch_dtype=dtype,
        )
        modelo.eval()
        nome = referencia if local else f"{referencia}@{revisao}"
        return cls(tokenizer, modelo, nome_modelo=nome, lote=lote)

    def _id_unitario(self, rotulo: str) -> int:
        ids = self.tokenizer.encode(rotulo, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"rótulo {rotulo!r} não é um token unitário para este tokenizer")
        return int(ids[0])

    @staticmethod
    def _mensagens(trecho: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": f"<trecho>\n{trecho}\n</trecho>\nResposta:"},
        ]

    def classificar_muitos(self, trechos: Sequence[str]) -> list[ResultadoVerificacao]:
        if not trechos:
            return []
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("verificação LLM requer torch") from exc

        resultados: list[ResultadoVerificacao] = []
        for inicio in range(0, len(trechos), self.lote):
            parte = trechos[inicio:inicio + self.lote]
            prompts = [
                self.tokenizer.apply_chat_template(
                    self._mensagens(trecho), tokenize=False, add_generation_prompt=True
                )
                for trecho in parte
            ]
            entradas = self.tokenizer(prompts, padding=True, return_tensors="pt")
            dispositivo = self.modelo.get_input_embeddings().weight.device
            entradas = {chave: valor.to(dispositivo) for chave, valor in entradas.items()}
            with torch.inference_mode():
                logits = self.modelo(**entradas).logits[:, -1, self.ids_rotulos].float().cpu().tolist()
            resultados.extend(_resultado_de_logits(item, modelo=self.nome_modelo) for item in logits)
        return resultados
