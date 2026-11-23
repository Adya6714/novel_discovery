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
cost of the two strategies?** In the controlled CoT studies we found (2609.33134, 2602.01017), reading was the cheaper strategy; the reverse trade-off is known from imitation learning (the "copycat" problem: de Haan et al. 2019; Wen et al. 2020), and the general principle that among redundant features training prefers the more *available* one is established (Hermann & Lampinen 2020; Hermann et al. 2023; Shah et al. 2020). Our contribution is to test that principle, with exact interventions and an independently measured cost, on written chain-of-thought intermediates. We construct a task family in
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

## 4. Task (as actually run; see decision log D14–D16 for the history)
Inputs x_1..x_12 ~ U(Z_11) (fixed length L = 12). Written chain c_j = Σ_{i=max(1,j−K+1)}^{j} x_i (mod 11).
Interleaved sequence x_1 c_1 x_2 c_2 … x_12 c_12, inputs as tokens 0–10 and chain values as tokens 11–21; teacher-forced loss on chain
tokens only. The prediction of c_{j+1} is made at the position of x_{j+1} (the query token itself). Two exact strategies exist:

| strategy | formula | operands retrieved by attention, j ≥ K | j < K |
|---|---|---|---|
| READ | c_j + x_{j+1} − x_{j+1−K} | 2: c_j (offset −1), x_{j+1−K} (offset −2K) | 1: c_j |
| RECOMPUTE | x_{j+2−K} + … + x_{j+1} | K − 1: x_{j+2−K} … x_j (offsets −2, −4, …) | j |

K = 1: the chain is useless (c_j = x_j). K = 2: RECOMPUTE retrieves 1 token, READ 2. K = 3: parity (2 vs 2). K ≥ 4: READ cheaper by operand count.
K = 12: ordinary prefix-sum chain (no leaving element). **Operand count is only a heuristic; the cost claim is tested with an independent,
operational measure of cost (§7.1).** Known redundancies: c_1 = x_1, and c_j are prefix sums for j < K, so at j = K the leaving element
x_1 can be taken from the written c_1 (we observe this; §6.3). The primary outcome is therefore computed over j > K.

## 5. Methods
* **Model:** decoder-only pre-LN transformer, d = 128, 4 heads, MLP 4d, learned absolute positions, depth 2.
* **Training:** AdamW (lr 1e-3, wd 0.01), batch 256, one-cycle schedule, 3000 steps, fresh random data each step. A gate-and-continue
  procedure (blocks of 3000 steps, cap 18,000) was in place but never triggered: every model passed the gate at 3000 steps.
* **Gate:** teacher-forced accuracy ≥ 0.98 at every slot (L = 12, n = 2000).
* **Interventions (n = 2000 sequences per slot, j = 1..11), three independent kinds:**
  - *single chain corruption*: c_j ← r ≠ c_j; FOLLOW = prediction equals READ(r); RECOMPUTE = prediction equals true c_{j+1}.
  - *coherent counterfactual chain*: c_1..c_j ← chain of redrawn inputs x'_1..x'_j (rows with c'_j = c_j dropped). Under it, a model that
    recomputes from the visible inputs scores RECOMPUTE; reading an *older* chain token lands in OTHER (not FOLLOW), so this intervention
    is used only to exclude older-chain reading for models that score RECOMPUTE.
  - *input-side*: x_j ← x'_j with the written chain untouched; READ predicts the original c_{j+1}, RECOMPUTE predicts c_{j+1} + (x'_j − x_j).
  - All three are validated on hand-coded exact READ / RECOMPUTE oracles (`test_oracle.py`: each oracle scores 1.000 on its own label at
    every slot). Classification is by argmax, so it identifies the *dominant* route; probabilities are also stored.
* **Forced-strategy models (independent cost):** identical models whose attention is restricted by a whitelist so that only READ
  (query sees itself, c_j, x_{j+1−K}) or only RECOMPUTE (query sees x_{j+2−K}..x_{j+1}) is computable; chain-token positions see only
  themselves. Cost := steps until every slot reaches ≥ 0.99 accuracy (evaluated every 100 steps) and the area under the worst-slot
  accuracy curve. 2 seeds × K ∈ {2, 3, 4, 6}.
* **Position-offset control:** models trained with a random absolute-position shift (0–16) per batch, evaluated at shifts 0, 8, 16.
* **Census:** per slot label READ (FOLLOW ≥ 0.9), RECOMPUTE (RECOMPUTE ≥ 0.9) or MIXED.
* **Statistics:** seed (model) is the unit; per-slot n = 2000 bounds within-model SE by 0.011 but is not replication. Proportions across
  models are given with exact (Clopper–Pearson) 95 % intervals.

## 6. Results
(filled in below by section)

## 10. Lonely-runner secondary study
See `lonely_runner/RESULTS.md`.
