# Guardrail Metrics

Populated after running the adversarial suite. Target: detection rate >95% per layer, FPR <5%.

## Adversarial Suite Results

| Layer | Description | True Positives | False Negatives | Detection Rate | False Positives | FPR |
|-------|-------------|---------------|-----------------|---------------|-----------------|-----|
| 1 | Read-only / single-statement | TBD | TBD | TBD | TBD | TBD |
| 2 | Table + column allowlist | TBD | TBD | TBD | TBD | TBD |
| 3 | PHI-column hard block | TBD | TBD | TBD | TBD | TBD |
| 4 | Small-cell suppression (k=11) | TBD | TBD | TBD | TBD | TBD |
| 5 | Prompt-injection detection | TBD | TBD | TBD | TBD | TBD |

Run with:
```bash
pytest src/ehrcopilot/guardrails/tests/adversarial_suite.py -v --tb=short
```
