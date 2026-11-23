# Preregistration: Do chain-trained transformers read or recompute their written intermediates?

Frozen: 2026-11-23, **before** any window-task model was trained or evaluated. Only a group-word-problem pilot
(Z60 vs A5, 2 layers, see `cot_reliance/logs/pilot_*`) had been launched; its results were not yet available when this was
written. Any later change to this protocol is logged in `docs/05_decision_log.md` with its reason.

## Primary research question
When a transformer is trained from scratch, with teacher forcing, on *correct* chains of written intermediate values, and the
next intermediate can be produced either by READING the previous written intermediate or by RECOMPUTING it from the input,
which strategy does training select — and is the selection governed by the relative computational cost of the two strategies?

## Task
Inputs x_1..x_L ~ Uniform(Z_p), p = 11, L ~ Uniform{2..12} per batch. Written chain c_j = Σ_{i=j-K+1..j} x_i mod p
(window sum, K ∈ {1,2,3,4,6,8,12}; K = 12 ≥ L is the ordinary prefix-sum chain-of-thought).
READ: c_{j+1} = c_j + x_{j+1} − x_{j+1−K} (3 operands). RECOMPUTE: Σ of the last min(K, j+1) inputs.
Both are exactly correct on all training data, so the training loss cannot distinguish them.

## Primary measurement (causal intervention)
At evaluation length L = 12, for chain slot j ∈ {1..11}, overwrite the written c_j with r ≠ c_j (uniform), and record the model's
greedy prediction for c_{j+1}:
* FOLLOW rate: prediction = r + x_{j+1} − x_{j+1−K} (mod p)
* RECOMPUTE rate: prediction = true c_{j+1}
* OTHER.
Primary outcome per model: mean FOLLOW rate over slots j ≥ K (where READ and RECOMPUTE operand counts are 3 vs K)
and separately over j < K. n = 2000 sequences per slot.

Secondary: (a) consistent counterfactual chain (all of c_1..c_j rewritten from redrawn inputs x'_1..x'_j) to rule out
reading of *older* chain tokens; (b) free-running continuation; (c) probability mass on FOLLOW vs RECOMPUTE tokens;
(d) "direct" models trained without chain, to measure in-pass computability of a K-term sum.

## Hypotheses
* **H1 (cost-tracking selection):** FOLLOW rate increases with K; for K ≤ 2 models predominantly RECOMPUTE (FOLLOW < 0.5),
  for K ≥ 6 predominantly FOLLOW (> 0.5).
* **H0a (always read):** FOLLOW > 0.9 for all K ≥ 2 (reading the immediately preceding token is always preferred).
* **H0b (always recompute when possible):** FOLLOW < 0.1 whenever the direct model can compute the K-sum.
* **H2 (position effect):** for fixed K, FOLLOW is lower at slots j < K (where RECOMPUTE needs only j+1 operands) than at j ≥ K.
* **H3 (depth):** increasing layers from 2 to 4 increases RECOMPUTE for intermediate K (more in-pass capacity).

## Disconfirmation criteria
H1 is rejected if FOLLOW is not monotone non-decreasing in K within ±0.1 across seeds, or if the K = 2 and K = 8 FOLLOW rates do
not differ by > 0.3 (seed-averaged). H0a is supported if all K ≥ 2 give FOLLOW > 0.9. Results are only interpreted for models
with clean teacher-forced chain accuracy ≥ 0.98 at L = 12 (models below this are reported but excluded from inference).

## Design / controls
* 3 seeds per (K, depth) cell; depths 2 and 4; d_model 128, 4 heads, AdamW lr 1e-3, wd 0.01, batch 256, one-cycle schedule;
  steps fixed per cell (chosen in a pilot for convergence and logged).
* Positive control for FOLLOW: K = 12 prefix chain (READ is 2-operand vs RECOMPUTE up to 12-operand) — expected to follow.
* Positive control for RECOMPUTE: K = 1 (c_j = x_j; READ is impossible to use, so FOLLOW must be ≈ 0).
* Negative control on metric: an untrained model gives FOLLOW ≈ RECOMPUTE ≈ 1/p.
* Statistics: per-model rates are binomial with n = 2000 per slot (SE ≤ 0.011); inference across seeds uses seed as the unit
  (mean ± SD, n = 3). No p-value hunting; the hypotheses above are the only confirmatory tests.

## Exclusions
Models failing the accuracy criterion; no other exclusions.
