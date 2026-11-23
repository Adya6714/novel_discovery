# Read or recompute? Relative computational cost governs whether chain-trained transformers use their own written intermediates

*Full research report.* Autonomous AI research programme, 2026-11-23. Repository: https://github.com/Adya6714/novel_discovery
(branch `research/discovery-programme`). Status labels used throughout: **[done]** = executed and verified here,
**[proposed]** = not executed.

<!-- RESULTS_PLACEHOLDER: sections 6-9 are filled from results/ by the analysis scripts; see bottom of file for provenance -->

## Abstract
(Written last; see section 12.)

## 1. Introduction
Chain-of-thought (CoT) traces are increasingly used as a window into model computation, for example in AI-safety monitoring. That
use presupposes that the written intermediates are causally used by the model. Two recent lines of work seem to disagree. On the one
hand, for tasks that a constant-depth network cannot solve in one pass ("serial" tasks), the chain is provably necessary
(Ceiling of a Task, arXiv 2609.33134), and small transformers trained on correct, noise-free chains behave step-wise faithfully
(arXiv 2602.01017). Even on a task that *could* be computed in one pass (the abelian word problem Z60), chain-trained models collapse
when their chain is erased (2609.33134). On the other hand, Liu et al. (2022) argued that a transformer given a scratchpad "can simply
ignore the scratchpad positions and find the same parallel shortcut", and corruption-trained models can perform well with meaningless
traces (arXiv 2505.13775).

These observations are compatible if *which* strategy a model learns depends on the task. Under teacher forcing on correct chains, a
model that reads its previous written intermediate and a model that recomputes the next intermediate from the input achieve identical
(zero) loss, so the training objective is indifferent; only the inductive bias of architecture + optimiser decides. We ask a narrow,
falsifiable question about that inductive bias: **is the choice between reading and recomputing governed by the relative computational
cost of the two strategies?** In every prior controlled study we found, reading was the cheaper strategy. We construct a task family in
which a single parameter K moves the cost of recomputation from below to above the cost of reading, and measure causal reliance on the
written chain with exact interventions.

## 2. Background and closest work
See `docs/03_novelty_matrix.md` for the comparison table (10 works). In brief: (i) necessity results constrain only above-ceiling tasks;
(ii) the step-wise vs skip-step transition of 2602.01017 is driven by *noise* in the training chains, and their complexity manipulation
always keeps reading simpler; (iii) Liu et al.'s statement about ignoring scratchpads was not measured causally; (iv) corruption studies on
LLMs are confounded by answer placement (2605.10799) and lack ground truth about which algorithm the model could use.

## 3. Research question and hypotheses (preregistered)
Preregistration: `docs/02_preregistration_cot_reliance.md` (frozen before any window-task model was trained). Deviations: decision log D10–D14.
* H1: FOLLOW rate increases with K; K ≤ 2 → mostly RECOMPUTE; K ≥ 6 → mostly FOLLOW.
* H0a: always read (FOLLOW > 0.9 for all K ≥ 2). H0b: always recompute when computable.
* H2: for fixed K, FOLLOW lower at slots j < K than j ≥ K. H3: depth increases RECOMPUTE at intermediate K.

## 4. Task
Inputs x_1..x_L ~ U(Z_11), L ~ U{2..12}. Written chain c_j = Σ_{i=max(1,j−K+1)}^{j} x_i (mod 11). Sequence: x_1..x_L, SEP, c_1..c_L;
loss on chain tokens only (teacher forcing). At chain slot j (the model's input token is c_j; it predicts c_{j+1}) the two exact
strategies are:

| strategy | formula | operands (j ≥ K) | operands (j < K) |
|---|---|---|---|
| READ | c_j + x_{j+1} − x_{j+1−K} | 3 (current token + 2 inputs) | 2 |
| RECOMPUTE | x_{j+2−K} + … + x_{j+1} | K | j + 1 |

K = 1 makes reading useless (c_j = x_j); K = 2 makes recomputing cheaper (2 vs 3); K = 3 equal (3 vs 3); K ≥ 4 makes reading cheaper;
K = 12 ≥ L is the ordinary prefix-sum CoT (reading = 2 operands, no leaving element).

## 5. Methods
* **Model:** decoder-only pre-LN transformer, d = 128, 4 heads, MLP 4d, learned absolute positions, depth 2 (and 4 where stated).
* **Training:** AdamW (lr 1e-3, wd 0.01), batch 256, one-cycle schedule, 4000 steps; fresh random data every step (no repetition).
  Models failing the accuracy gate were *continued* in blocks of 6000 steps (fresh data stream, new one-cycle schedule) up to 22,000 total
  steps (decision log D13); continuation history stored in each checkpoint's cfg.
* **Gate:** teacher-forced next-chain-token accuracy ≥ 0.98 at *every* slot for L = 12 (n = 2000). Only gated models enter inference.
* **Interventions (L = 12, n = 2000 sequences per slot, slots j = 1..11):**
  - *single*: overwrite c_j with r ≠ c_j uniformly; classify greedy prediction of c_{j+1} as FOLLOW (= READ applied to r),
    RECOMPUTE (= true c_{j+1}) or OTHER (FOLLOW and RECOMPUTE tokens always differ since r ≠ c_j).
  - *consistent*: overwrite c_1..c_j with the chain of independently redrawn inputs x'_1..x'_j (the written chain is internally coherent but
    contradicts the visible input); rows where the counterfactual c'_j equals c_j are dropped. A model that reads *any* chain token follows;
    only input-recomputation ignores it.
  - Probability mass on FOLLOW / RECOMPUTE tokens; free-running continuation to slot L and its agreement with the corrupted vs true final value.
* **Attention signature (no intervention):** layer-wise head-averaged attention from the slot-j query to x_{j+1} (used by both),
  x_{j+1−K} (used only by READ) and the window interior x_{j+2−K}..x_j (used only by RECOMPUTE).
* **Negative control:** untrained networks. **Positive controls:** K = 1 (must recompute) and K = 12 (prefix chain).
* **Statistics:** per-model binomial rates (n = 2000 per slot, SE ≤ 0.011); seed is the unit for cross-model inference (mean ± SD, 3 seeds).

## 6. Results
(filled in below by section)

## 10. Lonely-runner secondary study
See `lonely_runner/RESULTS.md`.
