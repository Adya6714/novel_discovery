# Publication-readiness assessment, failed hypotheses, future experiments

## What has been established (and how firmly)
| Claim | Evidence | Strength |
|---|---|---|
| C1. Transformers trained on *correct* written chains can ignore them entirely: at K ≤ 2, prediction of each next intermediate is recomputed from the input and does not depend on the written intermediate | 6/6 models; three independent intervention types; coherent counterfactual excludes older-chain reading; oracle-validated metric; reviewer reproduced | Strong (for this setting) |
| C2. At K ≥ 4 (incl. ordinary prefix CoT) the same architecture reads its written intermediates (errors propagate) | 15/15 models, all three interventions | Strong (for this setting) |
| C3. The selected route is the one that is more learnable in isolation (independent cost) | forced-strategy learnability, 3 seeds × K ∈ {2,3,4,6}; 9/9 decisive-K free models match | Moderate (4 K values, one architecture) |
| C4. Near cost parity (K = 3) selection is seed-dependent | 9 seeds (see REPORT §7.2) | Moderate; exploratory, post hoc |
| C5. "Self-correction" of corrupted intermediates is a by-product of route selection | free-run continuation, all models | Strong (descriptive) |
| L1. Lonely-runner amended conjecture holds in extended radii (n=7 ≤ 100, n=8 ≤ 70, n=9 ≤ 55); new k=2 septuple (2,7,9,11,13,20,54) | exact search, cross-checked | Strong, low significance |

## Novelty (after adversarial review)
The underlying principle (training selects the more available of redundant zero-loss solutions) is known (availability/simplicity
bias; copycat problem). New: its demonstration for written chain-of-thought intermediates with exact causal interventions, a single
cost dial that crosses the point where recomputation becomes cheaper, an independently measured cost that predicts the choice, and the
observation of seed-dependent selection at parity. To our knowledge no prior work shows that correct, noise-free chain supervision can
yield a model that ignores its chain.

## Realistic venue
Workshop paper (mechanistic interpretability / reasoning / CoT-faithfulness workshops at NeurIPS/ICLR/ICML, BlackboxNLP) as is.
TMLR becomes realistic with: (i) a second architecture (RoPE/ALiBi), (ii) depth and width variation, (iii) a second task family where cost
is manipulated without changing K (e.g., code the chain through a permutation or put filler between chain tokens), (iv) ≥ 20 seeds at
parity. The lonely-runner data are suitable only as a data note / appendix.

## Failed or abandoned hypotheses and directions (preserved)
1. **Group word problem pilot (Z60 vs A5)** — abandoned: slow learning; later found to replicate the design of arXiv 2609.33134 (D8).
2. **v1 layout (x…x SEP c…c, variable L)** — abandoned: leaving-element offset depends on L, making subtraction unlearnable in budget;
   misleading per-batch loss (D14). Lesson: always gate on fixed-length per-slot accuracy.
3. **Target off-by-one bug** — found by the preregistered gate (D10).
4. **"Mosaic at parity" as a general phenomenon** — weakened: seed 1 shows no mosaic, seed 2 recomputes everywhere; reviewer showed the
   seed-0 mosaic slot uses a position-specific circuit (REPORT §7.2 for the position-offset control).
5. **H2 (position effect at j < K)** — not supported; artefact of c_1 = x_1.
6. **Operand count as the cost** — replaced by measured learnability; the reviewer showed alternative proxies (multiset lookup) predict
   the wrong crossover.
7. **Lonely runner mod-3 pattern for k-values** — refuted by n = 9 search.
8. **Programmes 1–7, 11–13** — not pursued for lack of laboratory/field resources; audited in `docs/01`.

## Future experiments [proposed, not executed]
1. Relative positional encodings (RoPE/ALiBi) and variable length in the interleaved layout.
2. Cost manipulation at fixed K: filler tokens between chain values (raises READ cost), inputs encoded as two tokens (raises RECOMPUTE cost);
   prediction: selection follows the forced-learnability ranking.
3. ≥ 20 seeds and width/weight-decay sweeps at K = 3 (phase diagram), plus long training (30k steps) to test strategy stability.
4. Depth-4 arm (H3) and moduli p ∈ {7, 17, 97}.
5. Path patching of the c_j → query attention heads to move from dominant-route to mechanism-level claims.
6. Serial (above-ceiling) analogue: does the same selection principle decide *which part* of a long chain is read?
7. Lonely runner: n = 10 search; use the Yi–Gill–Bu effective bound to turn bounded searches into complete classifications for n = 5.
