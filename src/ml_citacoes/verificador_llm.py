"""Verifica usando LLM se a citação capturada apenas pelo LoRA é realmente uma citação jurídica ou é um falso posito.
O modelo recebe somente o trecho candidato, a decisão usa os logits da próxima posição, dentre três rótulos 
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen

from .contratos import ResultadoVerificacao


ROTULOS = {"A": "CITACAO", "B": "INCERTO", "C": "NAO_CITACAO"}
PROMPT_VERSAO = "zero_shot_v2"

PROMPT_SISTEMA = """Você é um especialista em direito brasileiro e deve classificar o trecho fornecido, sem tentar completar ou corrigir seu conteúdo.

Responda com exatamente uma letra:
A = citação jurídica identificável;
B = incerto;
C = não é citação jurídica identificável.

Uma citação identificável aponta, no próprio trecho, para uma fonte jurídica específica, como norma, processo, precedente, súmula, tema ou decisão. Não é necessário confirmar que a fonte realmente existe.

Classifique como A quando houver ao menos uma destas formas de identificação:
- espécie normativa, marcador jurídico ou classe processual acompanhada de número ou identificador;
- súmula, tema ou precedente numerado;
- decisão descritiva que combine tribunal ou órgão, ano ou data e relator identificado;
- referência reconhecível mesmo com ruído de OCR, quebra de linha ou pontuação corrompida, desde que o trecho preserve a espécie ou classe e uma parte reconhecível do identificador, mesmo que alguns caracteres estejam corrompidos.

Não exija simultaneamente tribunal, relator, ano, classe e número. Uma decisão descritiva pode ser identificável pela combinação entre tribunal, data ou ano e relator, mesmo sem número processual.

Classifique como C números soltos, fragmentos, nomes isolados de tribunais ou órgãos e menções genéricas a jurisprudência, legislação, matéria ou orientação. Classifique como C quando o trecho não fala de nenhuma fonte específica, apenas remete, de forma difusa, à matéria ou à orientação de um tribunal."""

PROMPT_SHA256 = hashlib.sha256(PROMPT_SISTEMA.encode("utf-8")).hexdigest()


def _softmax_restrito(valores: Sequence[float]) -> list[float]:
    if len(valores) != 3 or any(not math.isfinite(float(v)) for v in valores):
        raise ValueError("são necessários três logits finitos")
    maior = max(float(v) for v in valores)
    exponenciais = [math.exp(float(v) - maior) for v in valores]
    total = sum(exponenciais)
    return [valor / total for valor in exponenciais]


def _resultado_de_logits(logits: Sequence[float], *, modelo: str) -> ResultadoVerificacao:
    probabilidades = _softmax_restrito(logits)
    letras = tuple(ROTULOS)
    maior = max(probabilidades)
    empatados = [i for i, valor in enumerate(probabilidades) if math.isclose(valor, maior)]
    indice = letras.index("B") if len(empatados) > 1 else empatados[0]
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


def _resultado_ollama(resposta: dict, *, modelo: str) -> ResultadoVerificacao:
    """Converte os top-logprobs do primeiro token em probabilidades A/B/C."""
    itens = resposta.get("logprobs") or []
    if not itens:
        raise RuntimeError("Ollama não retornou logprobs; atualize o Ollama")
    primeiro = itens[0]
    candidatos = [primeiro, *(primeiro.get("top_logprobs") or [])]
    por_letra: dict[str, float] = {}
    for item in candidatos:
        token = str(item.get("token", "")).strip()
        if token in ROTULOS and "logprob" in item:
            por_letra[token] = max(por_letra.get(token, -math.inf), float(item["logprob"]))
    ausentes = set(ROTULOS) - por_letra.keys()
    if ausentes:
        raise RuntimeError(
            "Ollama não incluiu todos os rótulos em top_logprobs: " + ", ".join(sorted(ausentes))
        )
    return _resultado_de_logits([por_letra[letra] for letra in ROTULOS], modelo=modelo)


class VerificadorOllama:
    """Backend local do Ollama, sem geração livre além do primeiro token."""

    def __init__(
        self,
        *,
        modelo: str = "llama3.1:8b",
        endpoint: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ):
        self.modelo = modelo
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _chat(self, trecho: str) -> dict:
        corpo = json.dumps({
            "model": self.modelo,
            "messages": VerificadorLlama._mensagens(trecho),
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 20,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 1, "seed": 260926},
        }).encode("utf-8")
        requisicao = Request(
            f"{self.endpoint}/api/chat", data=corpo,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(requisicao, timeout=self.timeout) as resposta:
                return json.loads(resposta.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"falha ao consultar Ollama em {self.endpoint}: {exc}") from exc

    def classificar_muitos(self, trechos: Sequence[str]) -> list[ResultadoVerificacao]:
        return [
            _resultado_ollama(self._chat(trecho), modelo=f"ollama:{self.modelo}")
            for trecho in trechos
        ]
