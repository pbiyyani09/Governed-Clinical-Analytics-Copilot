# Semantic Cache Metrics

Populated after running two passes over the EHRSQL dev set.

## Results

| Metric | Value |
|--------|-------|
| Cache hit rate (2nd pass, same role) | TBD |
| Cache hit rate (cross-role, should be 0%) | TBD |
| p50 latency — cache path (ms) | TBD |
| p95 latency — cache path (ms) | TBD |
| p50 latency — full agent path (ms) | TBD |
| p95 latency — full agent path (ms) | TBD |
| Estimated $/query saved (at $0.01/query proxy) | TBD |

## Methodology

1. First pass: run all EHRSQL dev questions → populates cache
2. Second pass: re-run same questions (same role) → measure hit rate
3. Third pass: re-run same questions (different role) → must be 0 hits

Role-collision test: pairs of identical questions with `role=clinician` and `role=researcher`
must never produce a cross-role hit.
