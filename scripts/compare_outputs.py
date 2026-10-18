"""Sanity-check general quality of an abliterated model against the original.

Generation runs one model per process, so only one copy of the weights is ever in memory:

    python scripts/compare_outputs.py generate --model google/gemma-4-12B-it --out outputs/eval/original.json
    python scripts/compare_outputs.py generate --model outputs/gemma-4-12B-it-heretic --out outputs/eval/heretic.json
    python scripts/compare_outputs.py report outputs/eval/original.json outputs/eval/heretic.json --out outputs/eval/compare.md
"""

import argparse
import json
import time
from pathlib import Path

PROMPTS = [
    (
        "coding",
        "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
        "that merges overlapping intervals. Include a docstring and three assert-based tests.",
    ),
    (
        "math",
        "A train leaves at 2:40 PM and travels 210 km at an average speed of 84 km/h. "
        "At what time does it arrive? Show your reasoning briefly.",
    ),
    (
        "logic",
        "Alice is taller than Bob. Bob is taller than Carol. Dana is shorter than Carol. "
        "Who is the second shortest? Explain in two sentences.",
    ),
    (
        "explanation",
        "Explain the difference between TCP and UDP in under 120 words.",
    ),
    (
        "email",
        "Write a short, polite email to a client explaining that their project delivery will be "
        "delayed by one week due to a supplier issue, and propose a call to discuss.",
    ),
    (
        "poem",
        "Write a four-line poem about autumn rain with an AABB rhyme scheme.",
    ),
    (
        "format",
        "List exactly three benefits of regular exercise as a bulleted list, "
        "with no introduction or conclusion.",
    ),
]


def generate(model_path: str, out_path: str, max_new_tokens: int):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="mps",
    )
    model.eval()

    results = []
    for name, prompt in PROMPTS:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy, so differences come from the weights, not sampling.
        )
        elapsed = time.perf_counter() - start

        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(
            {
                "name": name,
                "prompt": prompt,
                "response": response,
                "tokens": len(new_tokens),
                "seconds": round(elapsed, 1),
            }
        )
        print(f"[{name}] {len(new_tokens)} tokens in {elapsed:.1f}s")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps({"model": model_path, "results": results}, indent=2)
    )
    print(f"Wrote {out_path}")


def report(a_path: str, b_path: str, out_path: str):
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())

    lines = [
        "# Original vs. abliterated: ordinary prompts",
        "",
        f"- **A:** `{a['model']}`",
        f"- **B:** `{b['model']}`",
        "- Greedy decoding, no system prompt.",
        "",
    ]
    for ra, rb in zip(a["results"], b["results"]):
        lines += [
            f"## {ra['name']}",
            "",
            f"> {ra['prompt']}",
            "",
            f"### A ({ra['tokens']} tokens)",
            "",
            ra["response"],
            "",
            f"### B ({rb['tokens']} tokens)",
            "",
            rb["response"],
            "",
            "---",
            "",
        ]

    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=512)

    rep = sub.add_parser("report")
    rep.add_argument("a")
    rep.add_argument("b")
    rep.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "generate":
        generate(args.model, args.out, args.max_new_tokens)
    else:
        report(args.a, args.b, args.out)


if __name__ == "__main__":
    main()
