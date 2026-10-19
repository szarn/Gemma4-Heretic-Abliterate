# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure how the refusal direction shows up in the residual stream, before and after abliteration.

Extraction runs one model per process (through Heretic's own model loader and prompt sets), and
the summary is small enough to commit:

    python scripts/residual_geometry.py extract --model google/gemma-4-12B-it --out outputs/residuals/original.pt
    python scripts/residual_geometry.py extract --model outputs/gemma-4-12B-it-heretic --out outputs/residuals/heretic.pt
    python scripts/residual_geometry.py analyze outputs/residuals/original.pt outputs/residuals/heretic.pt \
        --out outputs/analysis/residual_geometry.json

Run from the repo root so Heretic reads config.toml. Residuals are taken at the first generated
token, as Heretic does: train-split means (400 prompts per set) define the refusal directions, and
the 100 held-out evaluation prompts per set are kept whole for the per-prompt statistics.
"""

import argparse
import json
import math
import sys
from pathlib import Path

CHECKPOINT = "checkpoints/google--gemma-4-12B-it.jsonl"
SELECTED_TRIAL = 143


def extract(model_path: str, out_path: str, batch_size: int):
    import torch
    from heretic.config import Settings
    from heretic.model import Model
    from heretic.utils import load_prompts, set_seed

    # Settings parses the command line too, so hand it only the overrides it should see.
    sys.argv = ["heretic", "--model", model_path, "--batch-size", str(batch_size)]
    settings = Settings()
    set_seed(settings.seed)
    torch.set_grad_enabled(False)

    model = Model(settings)
    data = {"model": model_path}
    for key, spec, mean in (
        ("good_train", settings.good_prompts, True),
        ("bad_train", settings.bad_prompts, True),
        ("good_test", settings.good_evaluation_prompts, False),
        ("bad_test", settings.bad_evaluation_prompts, False),
    ):
        prompts = load_prompts(settings, spec)
        if mean:
            print(f"[{key}] residual means...")
            data[f"{key}_mean"] = model.get_residuals_mean(prompts)
        else:
            print(f"[{key}] residuals...")
            data[key] = model.get_residuals_batched(prompts).cpu()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"Wrote {out_path}")


def selected_direction_index() -> float:
    # Optuna's journal sets each trial user attribute in its own entry (op_code 8).
    attrs = {}
    for line in Path(CHECKPOINT).read_text().splitlines():
        op = json.loads(line)
        if op["op_code"] == 8:
            attrs.setdefault(op["trial_id"], {}).update(op["user_attr"])
    for trial in attrs.values():
        if trial.get("index") == SELECTED_TRIAL:
            return trial["direction_index"]
    raise ValueError(f"trial {SELECTED_TRIAL} not found in {CHECKPOINT}")


def analyze(original_path: str, heretic_path: str, out_path: str):
    import torch
    import torch.nn.functional as F

    original = torch.load(original_path, map_location="cpu")
    heretic = torch.load(heretic_path, map_location="cpu")
    good, bad = original["good_train_mean"], original["bad_train_mean"]

    # Per-layer refusal directions exactly as heretic/main.py computes them
    # (orthogonalize_direction = true), then the selected trial's global direction r
    # interpolated between layers as in heretic/model.py, abliterate().
    directions = F.normalize(bad - good, p=2, dim=1)
    good_directions = F.normalize(good, p=2, dim=1)
    directions = F.normalize(
        directions - (directions * good_directions).sum(1, keepdim=True) * good_directions,
        p=2,
        dim=1,
    )
    direction_index = selected_direction_index()
    weight, index = math.modf(direction_index + 1)
    r = F.normalize(directions[int(index)].lerp(directions[int(index) + 1], weight), p=2, dim=0)

    # How well each layer's own direction separates held-out harmful from harmless prompts.
    good_proj = (original["good_test"] * directions).sum(-1)
    bad_proj = (original["bad_test"] * directions).sum(-1)
    d_prime = (bad_proj.mean(0) - good_proj.mean(0)) / ((bad_proj.var(0) + good_proj.var(0)) / 2).sqrt()

    def cosines(residuals):
        c = F.cosine_similarity(residuals, r.view(1, 1, -1), dim=-1)
        stats = {"mean": c.mean(0)}
        stats |= {name: c.quantile(q, dim=0) for name, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75))}
        return {k: v.tolist() for k, v in stats.items()}, c[:, -1].tolist()

    summary = {
        "direction_index": direction_index,
        "n_layers": good.shape[0] - 1,
        "hidden_size": good.shape[1],
        "note": "Layer index 0 is the embedding output; index l + 1 is the output of decoder layer l.",
        "refusal_norm_ratio": ((bad - good).norm(dim=1) / good.norm(dim=1)).tolist(),
        "direction_cosine_to_r": (directions @ r).tolist(),
        "separation_d_prime": d_prime.tolist(),
        "cosine_to_r": {},
        "final_layer_cosines": {},
    }
    for name, data in (("original", original), ("heretic", heretic)):
        summary["cosine_to_r"][name] = {}
        summary["final_layer_cosines"][name] = {}
        for label, key in (("harmless", "good_test"), ("harmful", "bad_test")):
            stats, final = cosines(data[key])
            summary["cosine_to_r"][name][label] = stats
            summary["final_layer_cosines"][name][label] = final

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=1))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ext = sub.add_parser("extract")
    ext.add_argument("--model", required=True)
    ext.add_argument("--out", required=True)
    ext.add_argument("--batch-size", type=int, default=32)

    ana = sub.add_parser("analyze")
    ana.add_argument("original")
    ana.add_argument("heretic")
    ana.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "extract":
        extract(args.model, args.out, args.batch_size)
    else:
        analyze(args.original, args.heretic, args.out)


if __name__ == "__main__":
    main()
