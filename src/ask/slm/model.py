from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass
class SLMOutput:
    text: str
    logits: torch.Tensor
    cost: float


class RandomSLM:
    """Drop-in SLM replacement that returns a uniformly sampled action token.

    Used as a "dice" baseline for ASK: it isolates the value of the language
    model's actual decisions from the value of the uncertainty gate. When
    ``generate`` is called, the prompt is ignored and a token from
    ``action_tokens`` is returned, so the rest of the ASK pipeline (MC dropout
    gate, Optuna τ search, IR/OR/valid-rate accounting) is unchanged.
    """

    def __init__(self, action_tokens: Iterable[str], seed: Optional[int] = 42):
        self._actions: List[str] = list(action_tokens)
        if not self._actions:
            raise ValueError("RandomSLM requires a non-empty action_tokens list")
        self._rng = _random.Random(seed)

    @property
    def actions(self) -> List[str]:
        return list(self._actions)

    def generate(self, prompt: str, decoding: Optional[Dict[str, Any]] = None) -> SLMOutput:
        text = self._rng.choice(self._actions)
        return SLMOutput(text=text, logits=torch.empty(0), cost=0.0)


def _is_qwen3x(model_name: str) -> bool:
    name = model_name.lower()
    return "qwen3" in name or "qwen3.5" in name


class HuggingFaceSLM:
    def __init__(self, model_name: str, device: str, dtype: str, thinking: bool = False):
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        # thinking=True: Qwen3/3.5 generates <think>…</think> before the action.
        # thinking=False: disables chain-of-thought (faster, needed when max_tokens is small).
        self._chat_template_kwargs: Dict[str, Any] = (
            {"enable_thinking": thinking} if _is_qwen3x(model_name) else {}
        )
        # Compile reduces per-call Python/CUDA overhead across 96k+ generate calls.
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self._warmup()

    def _warmup(self) -> None:
        dummy = self.tokenizer(["Hello"], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            for _ in range(3):
                self.model.generate(**dummy, max_new_tokens=5, do_sample=False,
                                    pad_token_id=self.tokenizer.pad_token_id)

    def generate(self, prompt: str, decoding: Dict[str, Any]) -> SLMOutput:
        system = decoding.get("system")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs,
        )
        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=decoding["max_tokens"],
                do_sample=False,
                repetition_penalty=decoding.get("repetition_penalty", 1.1),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_ids = generated_ids[:, input_len:]
        text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

        # Strip any leaked thinking block (safety fallback for Qwen3)
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()

        return SLMOutput(text=text, logits=torch.empty(0), cost=float(new_ids.numel()))


def load_slm(cfg: Dict[str, Any]):
    """Build an SLM instance from a config dict.

    Supported providers:
      - ``"hf"`` (default) — local HuggingFace causal-LM via ``HuggingFaceSLM``.
      - ``"random"`` — ``RandomSLM`` baseline; expects ``actions`` (list of
        valid action tokens) in ``cfg`` and an optional ``seed``.
    """
    provider = cfg.get("provider", "hf")
    if provider == "hf":
        return HuggingFaceSLM(
            model_name=cfg["model"],
            device=cfg.get("device", "auto"),
            dtype=cfg.get("dtype", "float16"),
            thinking=cfg.get("thinking", False),
        )
    if provider == "random":
        actions = cfg.get("actions")
        if not actions:
            raise ValueError("provider='random' requires cfg['actions']")
        return RandomSLM(actions, seed=cfg.get("seed", 42))
    raise ValueError(f"Unsupported provider: {provider}")
