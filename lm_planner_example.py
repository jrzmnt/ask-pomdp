"""
LM Planner Module
-----------------
Loads a language model from HuggingFace based on a YAML config and uses it
as an async planner within a PPO+SLM hybrid system.

Expected config.yaml structure:
    model:
        name: "Qwen/Qwen2.5-0.5B-Instruct"   # HuggingFace model ID
        device: "cuda"                          # "cuda" | "cpu" | "auto"
        dtype: "float16"                        # "float16" | "bfloat16" | "float32"
        max_new_tokens: 256
        temperature: 0.7
        do_sample: true
    prompts:
        planner:
            system: "You are a planning agent..."
"""

import asyncio
import logging
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..envs.env_utils import build_state, decode_obs
from ..envs.env_utils import ActionValidator, VALIDATORS
from ..envs.env_utils import PARSERS
from .safety_module import ACTION_DELTAS
from .base_lm_module import BaseLMModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LM Planner
# ---------------------------------------------------------------------------


class LMPlanner(nn.Module):
    """
    A PyTorch module that wraps a HuggingFace causal LM for trajectory planning.

    The forward pass is synchronous; async planning is handled by `plan_async`,
    which should be awaited from the orchestration loop.

    Args:
        config_path : Path to the YAML configuration file.
        parser      : Callable that converts an environment state dict to a
                      prompt string. Defaults to the FrozenLake parser.
        env_name    : Key into the PARSERS registry (used if parser is None).
    """

    def __init__(
        self,
        config_path: str,
        parser: Optional[Callable[[dict], str]] = None,
        validator: Optional[ActionValidator] = None,
        env_name: str = "frozen_lake",
    ) -> None:
        super().__init__()

        self.config = self._load_config(config_path)
        self.parser = parser or PARSERS[env_name]
        self.validator = validator or VALIDATORS.get(env_name, VALIDATORS["default"])

        model_cfg = self.config["model"]
        self._device_map = model_cfg.get("device", "cpu")
        self.device = BaseLMModule._resolve_device(self._device_map)
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self.temperature = model_cfg.get("temperature", 0.7)
        self.do_sample = model_cfg.get("do_sample", True)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(model_cfg.get("dtype", "float32"), torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            dtype=self.dtype,
            device_map=self._device_map,
        )
        self.model.eval()

        # Prompt config
        prompt_cfg = self.config.get("prompts", {}).get("planner", {})
        self.system_prompt = prompt_cfg.get("system", "You are a planning assistant.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _build_chat_prompt(self, user_message: str) -> str:
        """
        Format input using the model's chat template if available,
        otherwise fall back to a simple system/user format.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return (
            f"[SYSTEM] {self.system_prompt}\n" f"[USER] {user_message}\n" "[ASSISTANT]"
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def forward(self, state: dict) -> str:
        grid = state["grid"]
        size = state["size"]
        pos = state["position"]
        rows, cols = len(grid), len(grid[0])
        max_steps = size * size * 2

        plan = []
        visit_counts = {pos: 1}
        step_idx = 0

        while step_idx < max_steps:
            if grid[pos[0]][pos[1]] == "G":
                logging.debug(f"[Planner] Goal reached at step {step_idx}.")
                break

            current_state = {
                "grid": grid,
                "position": pos,
                "size": size,
                "history": list(plan),
                "visit_counts": dict(visit_counts),
            }

            action = None
            rejected = set()

            for retry_idx in range(5):
                current_state["rejected"] = rejected

                user_message = self.parser(current_state)
                prompt = self._build_chat_prompt(user_message)
                logging.debug(
                    f"[Planner] Step {step_idx} retry {retry_idx} at {pos} — prompt:\n{prompt}"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output_ids = self.lm_predict(inputs)

                new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
                raw = self.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
                if "</think>" in raw:
                    raw = raw.split("</think>", 1)[-1].strip()

                logging.debug(
                    f"[Planner] Step {step_idx} retry {retry_idx} at {pos} — raw: '{raw}'"
                )

                candidate = None
                for word in raw.upper().split():
                    word = word.strip("[](),.")
                    if word in ACTION_DELTAS:
                        candidate = word
                        break

                if candidate is None or candidate in rejected:
                    if candidate:
                        rejected.add(candidate)
                    continue

                dr, dc = ACTION_DELTAS[candidate]
                nr, nc = pos[0] + dr, pos[1] + dc

                action = candidate
                break

            if action is None:
                logging.debug(
                    f"[Planner] Step {step_idx} — all retries exhausted at {pos}, stopping."
                )
                exit()
                break

            dr, dc = ACTION_DELTAS[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            pos = (nr, nc)
            visit_counts[pos] = visit_counts.get(pos, 0) + 1
            plan.append(action)
            step_idx += 1

        logging.debug(f"[Planner] Final plan: {plan}")
        return "[" + ", ".join(plan) + "]"

    def lm_predict(self, inputs: dict) -> str:
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return output_ids

    # ------------------------------------------------------------------
    # Async interface (for PPO+SLM orchestration)
    # ------------------------------------------------------------------

    async def plan_async(self, state: dict) -> str:
        """
        Run planning in a thread pool so the RL loop is not blocked.

        Usage in orchestration loop:
            plan_task = asyncio.create_task(planner.plan_async(state))
            # PPO acts freely; awaits only when uncertain
            plan = await plan_task
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.forward, state)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    # Minimal example state for a 4x4 FrozenLake
    example_state = {
        "grid": [
            ["S", "F", "F", "F"],
            ["F", "H", "F", "H"],
            ["F", "F", "F", "H"],
            ["H", "F", "F", "G"],
        ],
        "position": (0, 0),
        "size": 4,
    }

    async def main():
        planner = LMPlanner(config_path="config.yaml", env_name="frozen_lake")

        # Async planning — PPO can act while this awaits
        plan_task = asyncio.create_task(planner.plan_async(example_state))

        print("PPO acting while LM plans...")
        # Simulate PPO doing something
        await asyncio.sleep(0)

        plan = await plan_task
        print(f"LM Plan: {plan}")

    asyncio.run(main())
