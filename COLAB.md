# Colab heavy-training workflow (RTX 6000 Pro 95 GB) + local 3090 inference

**Policy:** heavy/slow training that the local RTX 3090 (24 GB) can't do quickly
runs on the Colab RTX 6000 Pro (95 GB). Everything else — retrieval, evaluation,
and the **final product's inference** — runs on the local 3090. Gemma 3 12B at
4-bit is ~8 GB, so the fine-tuned adapter serves comfortably on the 3090.

## When to use Colab
- Multi-epoch SFT (e.g. 3 epochs ≈ 22 h on the 3090 vs a few hours on the 95 GB card).
- Larger models (gemma-3-27b) or full (non-LoRA) fine-tuning.
- Anything that OOMs or is too slow at bs=1 on 24 GB.

Light fine-tunes, the inference-time experiments, retrieval ablations, and serving
stay local.

## One-time setup
1. Local: `bash scripts/prepare_colab_bundle.sh` → produces `colab_bundle.tar.gz`
   (code + EHRSQL data + the built `mimic_iv_demo.db`, ~165 MB).
2. Upload it to Google Drive at `MyDrive/ehrcopilot/colab_bundle.tar.gz`.
3. In Colab: **Settings → Secrets → add `HF_TOKEN`** (your HF token; grants the
   gated Gemma models). Enable "Notebook access".

## Run
Open `notebooks/gemma_finetune_colab.ipynb` in Colab (on the 95 GB runtime) and
**Run all**. It:
1. mounts Drive and extracts the bundle to the **local Colab disk** (`/content/ehrcopilot`),
2. installs deps, reads `HF_TOKEN` from Colab Secrets,
3. auto-scales the batch size to the GPU (bs=16 on a 95 GB card),
4. runs **SFT (3 epochs) → ORPO pairs → Abstention-ORPO → quick eval**,
5. copies `checkpoints/{sft_gemma,orpo_gemma}` **back to Drive**.

## Back on the local 3090
Download `MyDrive/ehrcopilot/checkpoints/orpo_gemma/adapter_final` into
`checkpoints/orpo_gemma/adapter_final`, then evaluate / serve locally:

```bash
PYTHONPATH=src python -m ehrcopilot.eval.harness \
  data/ehrsql/ehrsql/mimic_iii/test.json \
  --model checkpoints/orpo_gemma/adapter_final \
  --few-shot data/ehrsql/ehrsql/mimic_iii/train.json \
  --retrieval-mode classifier --repair \
  --output tests/evalgen/gemma_orpo_full.json
```

Notes
- `qlora_sft` now takes `--batch-size` / `--grad-accum` so the big GPU is utilized;
  the 3090 path keeps bs=1 (defaults).
- The local 1-epoch run on the 3090 (in progress) gives a baseline; Colab is for
  the 3-epoch / larger runs if EX needs to climb past it.
