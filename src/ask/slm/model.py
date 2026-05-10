from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass
class SLMOutput:
    text: str
    logits: torch.Tensor
    cost: float


def _is_qwen3x(model_name: str) -> bool:
    name = model_name.lower()
    return "qwen3" in name or "qwen3.5" in name


class HuggingFaceSLM:
    def __init__(self, model_name: str, device: str, dtype: str):
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
        # Disable thinking mode for Qwen3/3.5 — without this the model generates
        # hundreds of <think> tokens before the answer, making inference ~10x slower.
        self._chat_template_kwargs: Dict[str, Any] = (
            {"enable_thinking": False} if _is_qwen3x(model_name) else {}
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


def load_slm(cfg: Dict[str, Any]) -> HuggingFaceSLM:
    if cfg.get("provider", "hf") != "hf":
        raise ValueError(f"Unsupported provider: {cfg['provider']}")
    return HuggingFaceSLM(
        model_name=cfg["model"],
        device=cfg.get("device", "auto"),
        dtype=cfg.get("dtype", "float16"),
    )
