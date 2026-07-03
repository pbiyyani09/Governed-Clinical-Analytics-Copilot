"""Quick local eval: compare two adapters on 100 test questions (80 ans + 20 unans).

Usage:
    python3 scripts/quick_eval_adapter.py --adapter final_output/omnisql_sft_adapter
    python3 scripts/quick_eval_adapter.py --adapter final_output/omnisql_orpo_adapter/omnisql_orpo_adapter
"""
import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ehrcopilot import config
from ehrcopilot.eval.harness import (
    load_ehrsql_split, post_process_sql, _exec_safe, results_match, ABSTAIN_TOKEN
)

ABSTAIN = ABSTAIN_TOKEN
SEED    = 42
N_ANS   = 80
N_UNANS = 20

def run_eval(adapter_path: str):
    import torch
    from unsloth import FastLanguageModel

    test = load_ehrsql_split(ROOT / "data/ehrsql2024/mimic_iv/test")
    rng  = random.Random(SEED)
    ans_ex   = [e for e in test if e.is_answerable]
    unans_ex = [e for e in test if not e.is_answerable]
    sample   = rng.sample(ans_ex, N_ANS) + rng.sample(unans_ex, N_UNANS)
    rng.shuffle(sample)

    print(f"\nLoading adapter: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=1536,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    def generate(questions):
        msgs = [
            [{"role": "system", "content": config.SYSTEM_PROMPT},
             {"role": "user",   "content": q}]
            for q in questions
        ]
        prompts = [tokenizer.apply_chat_template(m, tokenize=False,
                                                  add_generation_prompt=True)
                   for m in msgs]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1536).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        results = []
        for out in outs:
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            import re
            m = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
            results.append(m.group(1).strip() if m else text)
        return results

    correct_sql = wrong_sql = hallucinations = correct_abs = 0
    BATCH = 8
    t0 = time.time()

    for i in range(0, len(sample), BATCH):
        batch = sample[i:i+BATCH]
        preds = generate([ex.question for ex in batch])
        for ex, pred in zip(batch, preds):
            abstained = (pred.strip() == ABSTAIN)
            if ex.is_answerable:
                if abstained:
                    wrong_sql += 1
                else:
                    gr, ge = _exec_safe(post_process_sql(ex.gold_sql))
                    pr, pe = _exec_safe(post_process_sql(pred))
                    if pe is None and ge is None and gr and results_match(pr, gr):
                        correct_sql += 1
                    else:
                        wrong_sql += 1
            else:
                if abstained:
                    correct_abs += 1
                else:
                    hallucinations += 1

        done = i + len(batch)
        elapsed = time.time() - t0
        print(f"  [{done:3d}/100] correct={correct_sql} hall={hallucinations} "
              f"abs={correct_abs} {elapsed:.0f}s")

    total = len(sample)
    ex    = correct_sql / N_ANS
    rs10  = (correct_sql + correct_abs - 10 * hallucinations) / total

    print(f"\n{'='*50}")
    print(f"Adapter: {adapter_path}")
    print(f"  EX:              {ex:.4f}  ({correct_sql}/{N_ANS})")
    print(f"  RS(10):          {rs10:.4f}")
    print(f"  Hallucinations:  {hallucinations}/{N_UNANS}")
    print(f"  Correct abstain: {correct_abs}/{N_UNANS}")
    print(f"  Wrong SQL:       {wrong_sql}/{N_ANS}")
    print(f"  Time:            {(time.time()-t0)/60:.1f} min")
    return {"adapter": adapter_path, "ex": ex, "rs10": rs10,
            "hallucinations": hallucinations, "correct_abs": correct_abs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()
    run_eval(args.adapter)
