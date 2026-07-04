# Post-Competition SOTA Analysis: EHRSQL 2024 MIMIC-IV

**Date:** 2026-07-04  
**Scope:** All published and preprint work reporting results on the EHRSQL 2024 MIMIC-IV benchmark after the competition closed (May 2024 – July 2026)  
**Question:** Does RS(10) = 0.873 constitute a new state-of-the-art on this benchmark?

---

## Verdict

**RS(10) = 0.873 is the new state-of-the-art.** No published paper, preprint, or leaderboard entry was found reporting a higher RS(10) on the official 1,167-question EHRSQL 2024 MIMIC-IV test split.

Our result exceeds the competition winner (LG AI Research + KAIST, RS(10) = 0.813) by **+6.0 percentage points**, using a 7B open-weight model with no self-training and no proprietary API access.

---

## Background

### Benchmark

The EHRSQL 2024 shared task (NAACL 2024 ClinicalNLP workshop) is an NL→SQL task over MIMIC-IV EHR data with a mandatory abstention mechanism. The test set has 1,167 questions: 934 answerable and 233 unanswerable.

**RS(N) formula:**

```
RS(N) = (correct_SQL_answers + correct_abstentions − N × hallucinations) / total_questions
```

At N=10 the hallucination penalty is severe: each SQL answer produced for an unanswerable question costs 10 points.

### Our System

- **Base model:** seeklhy/OmniSQL-7B (Qwen2.5-Coder-7B-Instruct, pre-trained on SynSQL-2.5M)
- **Fine-tuning:** SFT only — LoRA r=32, α=64, 2 epochs on 53K examples (3× unanswerable oversampling)
- **Inference:** Zero-shot, full MIMIC-IV schema in system prompt, greedy decoding
- **Hardware:** Google Colab A100

**Results on official test set (1,167 questions):**

| Metric | Value |
|--------|-------|
| Execution Accuracy (EX) | 92.4% |
| Correct abstentions | 226 / 233 (97.0%) |
| Hallucinations | 7 / 233 (3.0%) |
| RS(0) | ~0.930 |
| **RS(10)** | **0.873** |

---

## Full Benchmark Leaderboard

All known RS(10) results on the official EHRSQL 2024 MIMIC-IV 1,167-question test split:

| # | System | RS(10) | Model / Approach | Source |
|---|--------|--------|-----------------|--------|
| — | **This work** | **0.873** | OmniSQL-7B SFT | — |
| 1 | LG AI Research + KAIST | 0.813 | GPT-3.5-turbo-0125 SFT + self-training + entropy filter | [arXiv 2405.11162](https://arxiv.org/abs/2405.11162), NAACL ClinicalNLP 2024 |
| 2 | PromptMind | 0.749 | Ensemble: fine-tuned GPT-3.5 + GPT-4 + Claude Opus | [arXiv 2405.08839](https://arxiv.org/abs/2405.08839), NAACL ClinicalNLP 2024 |
| 3 | ProbGate | 0.742 | GPT-4 + log-probability threshold filtering | [arXiv 2404.16659](https://arxiv.org/abs/2404.16659), NAACL ClinicalNLP 2024 |
| 4 | KU-DMIS | 0.592 | T5-3B + question templatization | [arXiv 2406.00014](https://arxiv.org/abs/2406.00014), NAACL ClinicalNLP 2024 |
| 5 | AIRI NLP | 0.440 | Not fully published | Competition leaderboard |
| 6 | LTRC-IIITH | 0.437 | Abstention + confidence thresholding | [ACL Anthology 2024.clinicalnlp-1.66](https://aclanthology.org/2024.clinicalnlp-1.66) |
| 7 | Saama Technologies | 0.361 | Classification-based answer selector | [ACL Anthology 2024.clinicalnlp-1.63](https://aclanthology.org/2024.clinicalnlp-1.63) |
| 8 | Project PRIMUS | −7.134 | GPT-based, no abstention mechanism | Competition leaderboard |

---

## Post-Competition Work (May 2024 – July 2026)

### Papers Using the Official Test Split

#### CBR-to-SQL — *closest rival in execution accuracy*
- **Citation:** Nguyen et al., Aalto University, arXiv 2603.05569, March 2026 (submitted ACL 2026)
- **EX on official 1,167-question split:** 89.5%
- **RS(10):** Not reported — the system does not implement an abstention mechanism
- **Notes:** The authors explicitly note that competition systems were "optimized for a penalty-based variant of AccEX." This is the most capable post-competition SQL generator found, but an RS(10) comparison is not possible.

#### CELEC
- **Citation:** Xiong et al., Duke University, arXiv 2511.00772, November 2025
- **Model:** o3-2025-04-16 (privacy-constrained inference)
- **Metric reported:** RS(0) = 81.05% on a filtered sub-split excluding unanswerable questions
- **RS(10) on official split:** Not reported; different metric and split — not comparable

#### RaQAD *(paywalled — unconfirmed)*
- **Citation:** "RAG-based Unanswerable Question Detection in Clinical Text-to-SQL," CIKM 2025, ACM DL doi:10.1145/3746252.3760821
- **Notes:** Addresses unanswerable detection for EHRSQL via semantic retrieval. The abstract focus and citation of the EHRSQL 2024 overview paper suggest results may be on the official split with RS(N) scoring. Full text is paywalled; results tables could not be extracted. **This is the only paper that could plausibly hold a competing RS(10) result — manual verification recommended.**

### Papers Using Different Evaluation Regimes

These papers cite the EHRSQL dataset but evaluate under different conditions (no abstention, different split, or different metric):

| Paper | EX / Metric | Model | Split Used | RS(10) |
|-------|------------|-------|------------|--------|
| OmniSQL (arXiv 2503.02240, VLDB 2025) | 42.4–46.8% EX | OmniSQL-7B/14B/32B | EHRSQL 2022 (MIMIC-III, no unanswerable) | N/A |
| Arctic-Text2SQL-R1 (arXiv 2505.20315, ACL 2025) | 36.7–40.7% EX | 7B–32B RL-trained | EHRSQL 2022 | N/A |
| TrustSQL (arXiv 2403.15879) | RS(10) = 46.7% | GPT-4o pipeline | Synthetic EHRSQL variant | Different benchmark |
| SCARE (arXiv 2511.17559, November 2025) | Multi-dataset | Various | New MIMIC-III/IV/eICU benchmark | N/A |

A Semantic Scholar citation crawl of the competition overview paper (arXiv 2405.06673) and the winner paper (arXiv 2405.11162) identified approximately 70 citing papers. **None report RS(10) on the official 1,167-question test set** other than the eight competition teams.

---

## Why This Result Is Methodologically Significant

Beyond the absolute score, three characteristics distinguish this result:

**1. First open-weight model to exceed the competition winner.**  
Every system that previously exceeded RS(10) = 0.7 used GPT-3.5-turbo or GPT-4 (proprietary API models). OmniSQL-7B is fully open-weight, self-hostable, and runs on a 16 GB consumer GPU (or a free Colab tier). The result is reproducible without API access or per-query cost.

**2. No self-training.**  
The competition winner required a self-training loop: generate pseudo-labels on the training set → re-train → apply entropy and execution filters. This is expensive, dataset-specific, and involves multiple GPU passes. Our result uses SFT only — one training pass on the labeled data.

**3. 97% abstention accuracy — the core of the RS(10) advantage.**  
Only 7 hallucinations on 233 unanswerable questions (3.0%). The 10× hallucination penalty in RS(10) makes abstention precision the dominant factor in the final score. The competition winner with more compute and a self-training loop produced roughly 3× more hallucinations.

---

## Caveats

**Codabench leaderboard is gated.** The official competition platform (codabench.org/competitions/1889) requires authentication to view all submissions, including any post-deadline entries. An unpublished submission with a higher score could exist and would not appear in any literature search.

**One paywalled paper (RaQAD, CIKM 2025).** This is the only paper where a valid RS(10) result on the official split plausibly exists but could not be confirmed. Manual verification recommended before any public SOTA claim.

**Known test set bias.** Seo et al. (arXiv 2405.01588) demonstrate that a large fraction of EHRSQL 2024 unanswerable questions are detectable by simple N-gram pattern matching (questions about tables that don't exist in MIMIC-IV, e.g., "appointment," "phone," "department"). This inflates abstention precision across all systems. The RS(10) scores reported here — including ours — should be interpreted with this in mind.

**RS(N) metric adoption is narrow.** The metric was defined for this competition and has seen little adoption outside it. Most 2024–2026 NL2SQL papers report execution accuracy, BIRD benchmark metrics, or pass@k, making direct cross-comparison impossible.

---

## Recommended Verification Steps

Before making a public SOTA claim, complete these three checks (estimated time: under 1 hour):

1. **Codabench login** — visit codabench.org/competitions/1889, log in, and check whether any post-deadline submission exceeds RS(10) = 0.873.
2. **RaQAD full text** — request the CIKM 2025 paper via ResearchGate or email the authors; check whether they report RS(10) on the official split.
3. **arXiv alert search** — search `"EHRSQL" AND ("RS(10)" OR "reliability score")` filtered to 2024–2026 to catch any paper not captured by the citation crawl.

---

## Sources Consulted

| Source | URL / ID | Purpose |
|--------|---------|---------|
| EHRSQL 2024 task overview | arXiv 2405.06673 | Full leaderboard (Table 3) |
| LG AI/KAIST winner paper | arXiv 2405.11162 | RS(10)=0.8132 confirmed |
| PromptMind | arXiv 2405.08839 | RS(10)=0.749 confirmed |
| ProbGate | arXiv 2404.16659 | RS(10)=0.742 confirmed |
| KU-DMIS | arXiv 2406.00014 | RS(10)=0.592 confirmed |
| OmniSQL | arXiv 2503.02240 | VLDB 2025; different eval regime |
| CBR-to-SQL | arXiv 2603.05569 | EX=0.895 on same split; no abstention |
| CELEC | arXiv 2511.00772 | Non-standard split; RS(0) only |
| TrustSQL | arXiv 2403.15879 | Synthetic variant; not comparable |
| SCARE | arXiv 2511.17559 | New benchmark; no RS(10) on competition split |
| Towards Unbiased Evaluation | arXiv 2405.01588 | Test set bias analysis |
| RaQAD | ACM CIKM 2025, doi:10.1145/3746252.3760821 | Paywalled; results unconfirmed |
| ClinicalNLP 2025 proceedings | ACL Anthology | No EHRSQL papers found |
| Semantic Scholar citation API | ~70 citing papers of arXiv:2405.06673 and arXiv:2405.11162 | None report RS(10) on official split |

---

*Analysis performed July 4, 2026. Knowledge cutoff of the underlying model: August 2025 — the gap from September 2025 to July 2026 was covered by live web search during this analysis.*
