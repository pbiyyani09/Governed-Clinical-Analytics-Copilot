# Colab Gemma 4 fine-tune + eval → local RTX 3090 inference

**Policy:** Colab (Pro+ GPU) does **fine-tuning and evaluation only**. The final/
production **inference runs on the local RTX 3090 (24 GB)**. The trained artifact is
a small LoRA adapter (~0.5 GB) that serves on the 3090.

## Model choice (constraint: must run inference on a 24 GB 3090)
Gemma 4, 4-bit inference footprint vs the 24 GB budget:

| model | HF id (trainable) | 4-bit | fits 3090? |
|---|---|---|---|
| **Gemma 4 12B** (recommended — largest that fits) | `unsloth/gemma-4-12b-it` | 6.7 GB | ✅ (4-bit; 8-bit 13.4 GB also fits) |
| Gemma 4 E4B (faster) | `unsloth/gemma-4-E4B-it` | 4.5 GB | ✅ |
| Gemma 4 E2B (fastest) | `google/gemma-4-E2B-it` | 2.9 GB | ✅ |
| Gemma 4 31B / 26B-A4B | (unsloth mirrors) | ~13–16 GB | ⚠️ tight/slow — not recommended |

**12B is the max** for comfortable 3090 inference; the notebook defaults to it.

## Drive flow (exactly your setup)
- Big (5 TB) account: hold the data; **share the folder `EHRSQL_GEMMA`** (with edit
  access) to the Colab Pro+ account.
- Colab account Drive (15 GB): folder **`ehrsql/`** contains a **shortcut** to
  `EHRSQL_GEMMA`. So in Colab the path is
  `/content/drive/MyDrive/ehrsql/EHRSQL_GEMMA/`.
- Files written back (LoRA adapters ~0.5–1 GB, eval JSON) are small → safely under
  the 15 GB Colab-account quota. (Do **not** save merged full models there.)

## Steps
1. **Local:** `bash scripts/prepare_colab_bundle.sh` → `ehrsql_gemma_bundle.zip`
   (code + EHRSQL data + the 109 MB `mimic_iv_demo.db`).
2. Upload the zip into the **shared `EHRSQL_GEMMA` folder** (via your 5 TB account).
3. **Colab:** add a Secret named `HF_TOKEN` (gated Gemma 4 access; enable notebook access).
4. Open **`notebooks/gemma4_finetune_eval_colab.ipynb`** → set `MODEL` (default
   `unsloth/gemma-4-12b-it`) → **Run all**. It:
   - mounts Drive, unzips the bundle to the **local Colab disk**,
   - installs latest Unsloth (Gemma 4 support), reads `HF_TOKEN`,
   - auto-scales batch to the GPU,
   - runs **SFT (3 epochs) → ORPO pairs → Abstention-ORPO → full 1786 eval**,
   - copies `checkpoints/{sft_gemma4,orpo_gemma4}` + the eval JSON **back to
     `EHRSQL_GEMMA/`**.
5. **Local 3090:** download `EHRSQL_GEMMA/checkpoints/orpo_gemma4/adapter_final`
   into `checkpoints/orpo_gemma4/` and run inference/serving (see notebook's last cell).

## Dependencies (important — Gemma 4 is bleeding-edge)
Gemma 4 is `model_type: gemma4_unified`, unsupported by the PyPI `unsloth` (which pins
`transformers<=5.5`). Install **Unsloth from git and let it choose its own transformers**
— do NOT pin transformers (a pin like `==5.10.1` conflicts with the git `unsloth_zoo`,
which currently wants `transformers 5.12.x`):

```bash
pip install "unsloth @ git+https://github.com/unslothai/unsloth" \
            "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo"
pip install datasets sentencepiece timm scikit-learn rank-bm25 sqlglot faiss-cpu einops
```
unsloth-from-git pulls compatible transformers/trl/peft/bitsandbytes/accelerate itself.
If you already ran a cell with the old transformers in the session: **Runtime → Restart**,
then run from the install cell.

## Notes
- **Use the 12B text model** (`unsloth/gemma-4-12b-it`, loaded via Unsloth `FastModel`).
  E2B/E4B are *multimodal* (need `FastVisionModel`) — not this text pipeline. 26B-A4B/31B
  are text but too big to serve on the 24GB 3090.
- `qlora_sft` takes `--batch-size`/`--grad-accum` (notebook scales them to VRAM).
- SFT is 3 epochs because 1 epoch plateaued EX ~0.40 locally; more epochs is the EX lever.
- The training code already handles Gemma's `token_type_ids` (ORPO) and markdown-fenced
  SQL (eval); the same code path that ran Gemma 3 locally runs Gemma 4 on Colab.
