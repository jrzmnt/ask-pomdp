"""
Diagnose SLM inference speed and thinking token usage.
Usage: python scripts/diagnose_slm.py --model qwen35-0.8b
"""
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "0.5b":        "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b":        "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-0.6b":  "Qwen/Qwen3-0.6B",
    "qwen3-1.7b":  "Qwen/Qwen3-1.7B",
    "qwen35-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen35-2b":   "Qwen/Qwen3.5-2B",
}

PROMPT = (
    "You are an agent navigating a grid world.\n"
    "Output EXACTLY one word: TURN_LEFT or TURN_RIGHT or FORWARD"
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--n-calls", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=5)
    args = p.parse_args()

    model_name = MODELS[args.model]
    print(f"\nModel: {model_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    ).eval()

    is_qwen3x = "qwen3" in model_name.lower()
    chat_kwargs = {"enable_thinking": False} if is_qwen3x else {}

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"Input tokens: {input_len}")

    print(f"\nRunning {args.n_calls} calls (max_new_tokens={args.max_tokens})...\n")

    times = []
    for i in range(args.n_calls):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        new_ids = out[:, input_len:]
        n_tokens = new_ids.shape[-1]
        raw = tokenizer.decode(new_ids[0])
        clean = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

        if i < 3:
            print(f"  [{i+1}] {elapsed*1000:.0f}ms | {n_tokens} tokens")
            print(f"       raw:   {repr(raw)}")
            print(f"       clean: {repr(clean)}")

    avg = sum(times) / len(times)
    projected_h = (avg * 96_000) / 3600
    print(f"\nAvg latency:      {avg*1000:.0f} ms/call")
    print(f"Projected (96k):  {projected_h:.1f}h")
    print(f"Thinking active:  {'YES — <think> in raw output!' if any('<think>' in tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)[:, input_len:][0]) for _ in range(1)) else 'no'}")

if __name__ == "__main__":
    main()
