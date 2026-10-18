# Gemma 4 12B-it abliteration with Heretic on Apple Silicon

Decensoring `google/gemma-4-12B-it` with [Heretic](https://github.com/p-e-w/heretic) 1.4.0 on an
M5 Max (48 GB unified memory, MPS, BF16), then converting to MLX 8-bit, Q8_0 GGUF, and Ollama.
Model weights are not in this repo (tens of GB); this is the recipe, settings, and study record.

## Result

| | Refusals (100 harmful prompts) | KL divergence vs. original |
|---|---|---|
| Original `google/gemma-4-12B-it` | 99/100 | 0 |
| Trial 143 (saved, final eval) | 37/100 | 0.0397 |

- 200 trials, default Heretic optimization settings, 200-token generations (`max_response_length = 200`,
  up from the default 100, because Gemma 4 sometimes refuses partway through a response).
- Refusals are Heretic's marker-based count: any response containing words like "illegal" or
  "disclaimer" counts, so compliant answers with a caveat are included. Only 3 of the 37 used explicit
  refusal wording.
- Ordinary-prompt quality is essentially unchanged; see [`outputs/eval/compare.md`](outputs/eval/compare.md).

Pareto front from the study (`checkpoints/google--gemma-4-12B-it.jsonl`):

| Trial | Refusals | KL |
|---|---|---|
| 143 | 35/100 | 0.0393 |
| 79 | 36/100 | 0.0376 |
| 114 | 45/100 | 0.0293 |
| 164 | 47/100 | 0.0288 |
| 92 | 61/100 | 0.0190 |

## Setup

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt      # torch 2.14.1 (MPS), heretic-llm 1.4.0, peft 0.21.2, mlx-lm
```

`pillow` and `torchvision` are required by `Gemma4UnifiedProcessor` but are not pulled in by `heretic-llm`.

## Run Heretic

Settings are in [`config.toml`](config.toml) (read from the working directory). Always launch through
[`patches/run_heretic.py`](patches/run_heretic.py), not the `heretic` command:

```sh
caffeinate -ims env KAGGLE_KERNEL_RUN_TYPE=1 PYTHONUNBUFFERED=1 \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 \
  .venv/bin/python patches/run_heretic.py < /dev/null > logs/optimize.log 2>&1
```

- `KAGGLE_KERNEL_RUN_TYPE=1` switches Heretic's interactive menus to plain numbered `input()`, so runs
  can be backgrounded and driven by piped answers.
- The study journal in `checkpoints/` lets Heretic resume, or skip straight to the results menu.

Save a trial (answers: show previous results, first Pareto entry, save locally, path):

```sh
printf '1\n1\n1\noutputs/gemma-4-12B-it-heretic\n' | env KAGGLE_KERNEL_RUN_TYPE=1 \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 \
  .venv/bin/python patches/run_heretic.py
```

Evaluate the saved model against the original (loads one model at a time):

```sh
.venv/bin/python patches/run_heretic.py --model google/gemma-4-12B-it \
  --evaluate-model outputs/gemma-4-12B-it-heretic --batch-size 32
```

Quality comparison on ordinary prompts: [`scripts/compare_outputs.py`](scripts/compare_outputs.py).

## Convert

1. **Restore the upstream config layout.** transformers 5.20 `save_pretrained` rewrites Gemma 4's config
   into a `per_layer_config` form (dropping `global_head_dim` / `num_global_key_value_heads`), which
   mlx-lm 0.32.2 misreads. Copy Google's original `config.json` over the saved one (transformers loads
   both identically):
   ```sh
   cp ~/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/*/config.json outputs/gemma-4-12B-it-heretic/
   ```
2. **MLX 8-bit:**
   ```sh
   .venv/bin/mlx_lm.convert --hf-path outputs/gemma-4-12B-it-heretic \
     --mlx-path outputs/gemma-4-12B-it-heretic-mlx-8bit -q --q-bits 8 --q-group-size 64
   ```
3. **Q8_0 GGUF** with llama.cpp commit `620d222` (Homebrew `llama.cpp` 0.7.0, build 11669). The converter
   needs transformers 5.x for Gemma 4 (llama.cpp issue #24460), so it gets its own venv:
   ```sh
   brew install llama.cpp
   git clone https://github.com/ggml-org/llama.cpp tools/llama.cpp && git -C tools/llama.cpp checkout 620d222
   python3.13 -m venv .venv-gguf && .venv-gguf/bin/pip install -r requirements-gguf.txt
   .venv-gguf/bin/python tools/llama.cpp/convert_hf_to_gguf.py outputs/gemma-4-12B-it-heretic \
     --outtype bf16 --outfile outputs/gguf/gemma-4-12B-it-heretic-BF16.gguf
   llama-quantize outputs/gguf/gemma-4-12B-it-heretic-BF16.gguf outputs/gguf/gemma-4-12B-it-heretic-Q8_0.gguf Q8_0
   ```
4. **Ollama** from the Q8_0 GGUF, using Ollama's built-in Gemma 4 renderer/parser (see [`Modelfile`](Modelfile)):
   ```sh
   ollama create gemma4-12b-heretic -f Modelfile
   ```
   Ollama 0.40's safetensors importer uses its MLX engine and only offers `int4/int8/nvfp4/mxfp4/mxfp8`,
   not `q8_0`, hence the llama.cpp route.

## Use

All three run at ~39 tokens/s on an M5 Max. They default to Gemma 4's thinking mode unless disabled:

```sh
ollama run gemma4-12b-heretic --think=false
.venv/bin/mlx_lm.generate --model outputs/gemma-4-12B-it-heretic-mlx-8bit \
  --chat-template-config '{"enable_thinking": false}' --prompt "..."
llama-cli -m outputs/gguf/gemma-4-12B-it-heretic-Q8_0.gguf --reasoning off -ngl 99
```

The MLX, GGUF, and Ollama builds are text-only; the BF16 safetensors keep the vision/audio towers.

## Apple Silicon gotchas

- **MPS `linalg.qr` is pathologically slow** on tall matrices in torch 2.14.1 (a 3840×10 QR takes > 60 s).
  Heretic's default `row_normalization = "full"` calls `torch.svd_lowrank` for every abliterated module,
  so trial 1 stalled for 20+ minutes. [`patches/mps_svd_lowrank.py`](patches/mps_svd_lowrank.py) runs
  `svd_lowrank` on CPU for MPS inputs (bit-identical results, ~0.1 s per matrix).
- **Killing a process stuck in an MPS kernel does not stop the kernel**; the GPU stayed at 100% until reboot.
- **Batch size 64 crashed the Mac**: 200-token generations at batch 64 reached a 40.6 GB footprint
  (MPS recommended max is 37.4 GB on 48 GB) and starved WindowServer into a watchdog reboot. Hence
  `max_batch_size = 32` and the MPS watermark caps (the low ratio must be ≤ the high ratio).
- PEFT issue #3129 (`Gemma4ClippableLinear`) does not apply: the `gemma4_unified` architecture used by
  the 12B model has no such module.

## License

The code in this repository is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE)
(`AGPL-3.0-or-later`), matching Heretic, which `patches/run_heretic.py` imports.

The model itself is not included. `google/gemma-4-12B-it` and models derived from it are under Gemma 4's
Apache-2.0 license.
