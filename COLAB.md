# Colab EHRSQL-2024 fine-tune + eval → local RTX 3090 inference

**Benchmark:** EHRSQL-2024 (ClinicalNLP@NAACL 2024), **MIMIC-IV Demo**, scored with the
**official Reliability Score** — apples-to-apples with the leaderboard **RS(10) = 0.8132**
(see `DATA_INTEGRITY_REPORT.md` for why we switched off the old MIMIC-III data).

**Policy:** Colab (Pro+ GPU) does **fine-tuning and evaluation only**. Final/production
**inference runs on the local RTX 3090 (24 GB)**. The artifact is a small LoRA adapter
(~0.5 GB); Gemma 4 12B is 6.7 GB at 4-bit, so it serves comfortably on the 3090.

## Data (no PhysioNet credentialing needed)
EHRSQL-2024 runs entirely on the **open** MIMIC-IV Demo v2.2; the organizers ship a
preprocessed `mimic_iv.sqlite`. We vendor it under `data/ehrsql2024/mimic_iv/` together with
the official `train/valid/test` splits and an **augmented** train (`train_aug/`).

- **Augmentation** (`ehrcopilot.finetune.augment_ehrsql2024`): re-instantiates the ~100
  official question templates with values sampled from the live SQLite (every generated SQL is
  execution-verified) + synthesizes unanswerable questions. Default ≈40k answerable + 3k
  unanswerable. Aggressive setting chosen to cover EX + the 34 unseen test templates.
- **SFT data** (`ehrcopilot.finetune.prepare_sft`): formats `train_aug` (+ valid unanswerables)
  into chat JSONL, oversampled to 20% abstention → `data/ehrsql2024/sft_train_aug.jsonl`
  (~44k lines; regenerated inside Colab to keep the bundle small).

## Model choice (must run inference on a 24 GB 3090)
| model | HF id | 4-bit | fits 3090? |
|---|---|---|---|
| **Gemma 4 12B** (recommended — max that fits) | `unsloth/gemma-4-12b-it` | 6.7 GB | ✅ |
| Gemma 4 E4B (faster) | `unsloth/gemma-4-E4B-it` | 4.5 GB | ✅ |

## Drive flow (unchanged)
- 5 TB account holds the data; share the folder **`EHRSQL_GEMMA`** to the Colab Pro+ account.
- Colab account Drive (15 GB) has folder **`ehrsql/`** with a **shortcut** to `EHRSQL_GEMMA`,
  so the path is `/content/drive/MyDrive/ehrsql/EHRSQL_GEMMA/`.
- Only small adapters (~0.5–1 GB) + eval JSON are written back → well under the 15 GB quota.

## Steps
1. **Local:** `bash scripts/prepare_colab_bundle.sh` → `ehrsql_gemma_bundle.zip` (~15 MB:
   code + 2024 splits + `train_aug` + `mimic_iv.sqlite`).
2. Upload the zip into the shared `EHRSQL_GEMMA` folder.
3. **Colab:** add a Secret `HF_TOKEN` (gated Gemma 4; enable notebook access).
4. Open **`notebooks/gemma4_finetune_eval_colab.ipynb`** → **Run all**. It:
   - mounts Drive, unzips to local disk, installs the Unsloth stack,
   - **forces `transformers==5.10.1`** (Gemma 4 = `gemma4_unified` needs ≥5.10; git
     `unsloth_zoo` caps ≤5.5.0, so `--no-deps --force-reinstall` overrides it — the pip
     "requires <=5.5" warning is expected),
   - regenerates the SFT JSONL, runs **SFT (1 epoch on ≈44k) → ORPO pairs → Abstention-ORPO**,
   - **evaluates both adapters with the official Reliability Score** (`ehrcopilot.eval.eval2024`),
   - copies adapters + RS results back to `EHRSQL_GEMMA/`.
5. **Local 3090:** download `EHRSQL_GEMMA/checkpoints/orpo_g4_2024/adapter_final` and run
   `python -m ehrcopilot.eval.eval2024 data/ehrsql2024/mimic_iv/test --model … --few-shot
   data/ehrsql2024/mimic_iv/train --retrieval-mode embed --repair`.

## Dependencies (Gemma 4 is bleeding-edge)
```bash
pip install "unsloth @ git+https://github.com/unslothai/unsloth" \
            "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo" \
            bitsandbytes accelerate datasets sentencepiece timm scikit-learn rank-bm25 sqlglot faiss-cpu einops
pip install --no-deps --force-reinstall "transformers==5.10.1"   # gemma4 needs >=5.10
```
If a same-cell `import transformers` still prints 5.5.0, that's the kernel's cached module —
**Runtime → Restart**; the `!python` training subprocess reads the on-disk 5.10.1 regardless.

## Notes
- **Scoring parity:** `ehrcopilot.eval.ehrsql2024_scoring` is vendored verbatim from the
  shared-task `scoring_program` (do not "improve" it). Sanity anchors: abstain-all ≈ 19.97
  (official baseline 20.0), perfect predictions ≈ 98.6% EX (the rest are non-deterministic
  `ORDER BY … LIMIT` ties — a property of the benchmark, not a bug).
- The training code handles Gemma's `token_type_ids` (ORPO) and the multimodal processor
  (`text=` kwarg) and markdown-fenced SQL (eval).
- 1 epoch over ~44k augmented examples ≈ the prior 3 epochs over the raw 5k in update count;
  bump `SFT_EPOCHS` to 2 for more. qlora_sft checkpoints every 250 steps (resumable).
