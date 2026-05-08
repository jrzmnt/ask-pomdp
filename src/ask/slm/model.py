from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SLMOutput:
    text: str
    logits: torch.Tensor
    cost: float


class HuggingFaceSLM:
    def __init__(self, model_name: str, device: str, dtype: str):
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str, decoding: Dict[str, Any]) -> SLMOutput:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=decoding["max_tokens"],
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        logits = outputs.scores[0][0]
        cost = float(generated_ids.numel())

        return SLMOutput(text=text, logits=logits, cost=cost)


def load_slm(cfg: Dict[str, Any]) -> HuggingFaceSLM:
    if cfg.get("provider", "hf") != "hf":
        raise ValueError(f"Unsupported provider: {cfg['provider']}")
    return HuggingFaceSLM(
        model_name=cfg["model"],
        device=cfg.get("device", "auto"),
        dtype=cfg.get("dtype", "float16"),
    )
