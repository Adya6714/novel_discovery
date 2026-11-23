# Read or recompute? Relative computational cost governs whether chain-trained transformers use their own written intermediates

*Full research report.* Autonomous AI research programme, 2026-11-23. Repository: https://github.com/Adya6714/novel_discovery
(branch `research/discovery-programme`). Status labels used throughout: **[done]** = executed and verified here,
**[proposed]** = not executed.


## Abstract
When a transformer is trained with teacher forcing on *correct* chains of written intermediate values, the loss cannot tell apart a model
that reads its previous written intermediate from one that recomputes the next intermediate from the input. We ask which route training
selects. We use window-sum chains whose window size K moves the relative cost of the two routes, and train 2-layer transformers from scratch
(24-model sweep plus 30 control models). We measure causal reliance with three oracle-validated interventions: chain-token corruption, coherent
counterfactual chains and input-side corruption. With correct chains and no noise, models with K ≤ 2 ignore their written intermediates
completely: corrupting the chain never changes the next prediction (6/6 models), and they silently "self-correct" corrupted chains.
Models with K ≥ 4, including ordinary prefix-sum chains, read their intermediates and propagate errors (15/15). An independent,
behaviour-free measure of cost — how fast each route is learned when the other is masked out — predicts the selected route whenever the
routes differ in learnability (K = 2, 4, 6; 3 seeds each). At near-parity (K = 3) the choice is seed-dependent: 6/9 models recompute, 2/9 read
and 1/9 splits by position. These split and read outcomes depend on learned absolute positions and disappear when positions are randomly
shifted during training, after which all 3/3 K = 3 models recompute, as predicted by learnability. The underlying principle — training prefers
the more available of redundant solutions — is known from shortcut learning and imitation learning. Our contribution is to show
it decides whether chain-of-thought intermediates are causally used, even when they are correct, below the serial "ceiling". Faithfulness
measured by corrupting intermediates is therefore a property of which route training selected, not of chain-of-thought supervision per se.
A secondary, independent study extends exhaustive exact searches of the Lonely Runner spectrum (n = 7 to speed 100, n = 8 to 70, n = 9 to 55),
finding no counterexample to the amended Fan–Sun conjecture and one new k = 2 septuple.

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

## 6. Results: free (unconstrained) models
All 24 depth-2 models (K ∈ {1,2,3,4,5,6,8,12} × 3 seeds) passed the gate (teacher-forced accuracy 1.000 at every slot); one (K=8, seed 2)
needed one continuation block (gate failed at 3000 steps, passed at 6000). Full per-model table: `cot_reliance/results/summary_table.md`.

### 6.1 Strategy switches with K (Fig. 1)
At slots j > K (K = 12: all slots):

| K | retrieved operands READ : RECOMPUTE | models READ-dominant / n | FOLLOW per seed (chain corruption) |
|---|---|---|---|
| 1 | — (chain useless) | 0/3 | 0.000, 0.000, 0.000 |
| 2 | 2 : 1 | 0/3 | 0.000, 0.000, 0.000 (RECOMPUTE 1.000 ×3) |
| 3 | 2 : 2 (parity) | 2/3 | 0.875, 0.968, **0.000** (seed 2: RECOMPUTE 1.000) |
| 4 | 2 : 3 | 3/3 | 1.000 ×3 |
| 5, 6, 8 | 2 : 4, 5, 7 | 3/3 each | 1.000 ×9 |
| 12 (prefix) | 1 : j | 3/3 | 0.999, 0.999, 0.997 |

The three intervention types (chain-token corruption, coherent counterfactual chain, input-side corruption) agree to within 0.006 for every
model, so the classification does not depend on how the model is perturbed. An untrained network scores FOLLOW 0.054 / RECOMPUTE 0.055 at K = 3
(chance ≈ 1/11 for each specific token, mass spread over tokens).

**Error propagation and self-correction.** In free-running continuation after a single corruption, READ-dominant models carry the error to
the final chain value (final value equals the corrupted-chain value in 100 % of sequences for K ≥ 4), whereas RECOMPUTE-dominant models (K ≤ 2,
K = 3 seed 2) return to the correct final value in 100 % of sequences. Whether a model "self-corrects" a corrupted intermediate is thus
entirely determined by which route it selected during training — not by any explicit error-detection ability.

### 6.2 Attention signatures
Averaged over slots, layer-0 attention concentrates on the window interior for RECOMPUTE models (K = 2 seed 0: 0.52 per interior position;
leaving element 0.004) and on the leaving element plus previous chain token for READ models (K = 3 seed 0: 0.39 / 0.34; K = 4: 0.41 / 0.40).
The adversarial reviewer showed, per head and per slot, that K = 3 seed 0's single RECOMPUTE slot (j = 6) uses a distinct, recompute-type
circuit (heads attend to offset −2 instead of −6 / −1). Averages are therefore used only as corroboration of whole-model labels, never
for per-slot claims.

### 6.3 Redundancy at j = K
At j = K the leaving element x_1 also exists as the written c_1 (c_1 = x_1). Under the coherent counterfactual, K = 4 seed 0 takes it from the
written chain in 85 % of sequences (from the input in 1 %); K = 3: 25 % vs 38 %. Models thus route operands through the written chain when
it duplicates them. This slot is excluded from the primary outcome.

## 7. Results: tests added after adversarial review
### 7.1 Independent cost: learnability of each strategy in isolation (Fig. 2)
Forced-READ and forced-RECOMPUTE models (attention whitelists), 3 seeds each, evaluated every 20 steps:

| K | steps to worst-slot acc ≥ 0.99, READ-only | RECOMPUTE-only | faster in isolation | free models READ-dominant |
|---|---|---|---|---|
| 2 | 200, 180, 200 | **80, 80, 80** | RECOMPUTE (≈ 2.4×) | 0/3 |
| 3 | 200, 180, 200 | 160, 180, 200 (to 0.5: 120 vs 140–160) | ≈ tie (RECOMPUTE marginally) | 2/3 |
| 4 | **180, 160, 180** | 400, 500, 560 | READ (≈ 2.8×) | 3/3 |
| 6 | **180, 160, 200** | not learned in 800 steps (acc 0.07–0.08) | READ | 3/3 |

Where one strategy is clearly more learnable in isolation, all free models select it (9/9 models at K ∈ {2, 4, 6}); at K = 3 the two are
nearly equally learnable, RECOMPUTE marginally faster early (to 0.5: 120 vs 140–160; AUC≤400: 0.759 vs 0.696), and 6/9 free models recompute
(§7.2). This operational cost is defined without reference to the free models'
behaviour, which answers the circularity objection; note that it measures cost *for this architecture and layout*, including factors
such as attention offsets that operand counting ignores.

### 7.2 Position-offset control and K = 3 census
Per-slot census (label READ if FOLLOW ≥ 0.9, RECOMPUTE if RECOMPUTE ≥ 0.9, else mixed); Fig. 3.

**K = 3, learned absolute positions, 9 seeds** (3 from the sweep + 6 added after review):

| pattern at slots j ≥ 3 | seeds | fraction (95 % CI) |
|---|---|---|
| RECOMPUTE at every slot | 2, 3, 4, 5, 6, 7 | 6/9 (0.30–0.93) |
| READ at every slot but ≤ 1 | 0 (slot 6 RECOMPUTE), 1 | 2/9 (0.03–0.60) |
| half/half mosaic (READ j = 3–5, RECOMPUTE j = 6–11) | 8 | 1/9 (0.003–0.48) |

At slot 2 (j < K: READ retrieves 1 operand vs 2) all 9 models READ; at slot 1 (tie, 1 vs 1) all 9 are mixed. The first two seeds we
trained happened to be the two READ seeds; the "read preference at parity" suggested by them (and by the reviewer, reasonably, from the
data then available) does not survive a larger sample.

**Position-offset control** (absolute positions randomly shifted by 0–16 during training; evaluated at shifts 0, 8, 16):
K = 3, 3 seeds: RECOMPUTE at every slot j ≥ 3, at every evaluation shift (9/9 model×shift cells), slot 2 RECOMPUTE or mixed, no mosaics;
K = 2: RECOMPUTE everywhere; K = 4: READ everywhere (all shifts). Conclusions: (i) the K = 3 READ seeds and mosaics depend on learned
absolute positions (reviewer objection 2.1 confirmed); (ii) the K = 2 / K = 4 contrast and the K = 3 majority choice do not; (iii) with
absolute-position shortcuts removed, free models choose the route that is more learnable in isolation at all tested K (2: RECOMPUTE,
3: RECOMPUTE, 4: READ). Forced-strategy learnability *with the same position shifts* (2 seeds): K = 3 — RECOMPUTE-only reaches 0.5 worst-slot accuracy
first (steps 120, 160 vs READ-only 200, 180; AUC over the first 400 steps 0.599 vs 0.563) but plateaus at 0.95 within 800 steps, whereas
READ-only reaches 0.99 at 400–440; K = 4 — READ-only faster on every criterion (0.99 at 340–400; RECOMPUTE-only reaches 0.5 at 300 and
plateaus at 0.83–0.86).

**Which cost criterion?** Across all six comparisons (absolute positions K = 2, 3, 4, 6; shifted K = 3, 4), *early* learnability (AUC over the first
400 steps, or steps to 0.5) predicts the free models' majority route in 6/6; steps-to-0.99 predicts 5/6 and fails for shifted K = 3. The criterion
was not fixed in advance (the analysis script computed steps-to-0.99 and AUC; steps-to-0.5 was added when the coarse runs proved too coarse).
We therefore state the result as: *the route that is learned faster early in training is the route the unconstrained model adopts*.

**Training dynamics** (K = 3, seed 7, checkpoints every 500 steps): at step 500 only slot 2 is solved (by READ, its cheapest route);
by step 1000 all slots are solved and the route pattern (mixed, READ, RECOMPUTE × 9) is final; it is unchanged at 1500–3000 steps.
No switching was observed within this budget (longer training untested).

### 7.3 Spillover test
Hypothesis (post hoc): READ circuits learned at slots j < K (where READ is strictly cheaper) spill over to parity slots. Test: train with
loss only on slots j > K (c_1..c_{K+1} unsupervised but present as input). Result: K = 3, 3/3 seeds RECOMPUTE at every supervised slot (standard
training: 6/9); K = 4 (1 seed) READ at every supervised slot. The direction is consistent with a small spillover effect, but 3/3 vs 6/9 is
not statistically distinguishable (Fisher exact two-sided p ≈ 0.5); we report the test as inconclusive. K = 4 READ does not depend on
early-slot supervision.

## 8. Hypotheses: verdicts
* **H1** (FOLLOW increases with K; K ≤ 2 RECOMPUTE, K ≥ 6 FOLLOW): **supported** (K ≤ 2: 0/6 READ; K ≥ 4: 15/15 READ; monotone except
  the seed-dependent parity point K = 3, where 6/9 models recompute). Disconfirmation criteria in the preregistration were not met.
* **H0a** (always read): **rejected** (K = 2: FOLLOW 0.000 in 3/3 models although the written chain is correct and usable).
* **H0b** (always recompute when computable): **rejected** (K ≥ 4 models read although forced-RECOMPUTE models show recomputation is learnable
  at K = 4).
* **H2** (lower FOLLOW at j < K): **not supported**; the j = 1 dip is explained by the c_1 = x_1 redundancy.
* **H3** (depth): **not tested** (depth-4 arm dropped, D17).
* Exploratory (post hoc, not preregistered): seed-dependent selection at parity; position-specific mosaics (§7.2).

## 9. Limitations
* Toy setting: one task family (window sums mod 11), one width (d = 128), depth 2, learned absolute positions, fixed length 12. Generality to
  other architectures, positional encodings, moduli and to LLM-scale CoT is untested.
* Constant-K tasks are below the "ceiling" of 2609.33134: the chain never bridges serial depth. The results speak to the below-ceiling regime,
  which is exactly where corruption-based faithfulness tests are ambiguous, not to serial tasks.
* "Cost" is operationalised as learnability under attention masks; masked models are not identical to the free model (they lack distractors).
* Argmax classification identifies the dominant route; we do not claim the minority route is absent.
* 3 seeds per K (9 at K = 3); training stopped ~1000 steps after zero error; long-training stability (grokking-style circuit change) untested.
* The preregistration was overtaken by post-pilot design changes (D14–D16); its timestamp is a local file time (first commit 07:21 UTC), not
  an external registry. All changes are logged with reasons.
* All interpretation and review were performed by AI agents; no human has yet checked the work.
## 10. Lonely-runner secondary study
Exhaustive, exact enumeration of near-tight tuples (ML < 1/n) with an independent integer kernel, validated against every count published by
Cordella (arXiv 2609.03444) and against brute-force exact ML at small radius. New radii: n = 7 ≤ 100 (1.6·10^10 tuples; 40 primitive
near-tight, k ∈ {1, 2}), n = 8 ≤ 70 (40, all k = 1), n = 9 ≤ 55 (13, all k = 1). No counterexample to the amended Fan–Sun conjecture; Kravitz's
original form holds for n = 8, 9 in range; new k = 2 septuple (2, 7, 9, 11, 13, 20, 54) with ML = 9/65. A conjectured mod-3 pattern for which
k occur was refuted by the n = 9 data. Significance is low after Yi–Gill–Bu (arXiv 2610.40694). Details: `lonely_runner/RESULTS.md`.

## 11. Implications
1. **Correct chain supervision does not guarantee chain use.** Below the serial ceiling, whether a chain-trained transformer uses its written
   intermediates is decided by which route is learned faster, not by the presence of a correct chain. Corruption-based faithfulness tests on such
   models measure that route choice.
2. **Identically trained models can differ in faithfulness.** At near-parity, the same data and architecture yield chain-readers and
   chain-ignorers depending on the seed, and on the positional encoding. Faithfulness evaluations should report variability across training runs.
3. **"Self-correction" can be an artefact of route selection.** Models that ignore their chain appear to correct injected errors perfectly;
   models that read propagate them. Neither involves error detection.
4. **Design lever.** To make a model depend on its chain (e.g., for monitorability), make reading more learnable than recomputing (for
   example by making the intermediate hard to recompute in one pass — the regime of 2609.33134); to make it robust to corrupted chains, the
   reverse.
These implications are demonstrated only in a 2-layer toy setting; transfer to large models is a hypothesis.

## 12. Reproducibility
All code, per-model results (`cot_reliance/results/*.json`), logs and figures are in the repository; checkpoints are regenerable
(seeds fixed; CPU training, ≈ 15 min per model on one core). `test_targets.py` and `test_oracle.py` are unit tests for data alignment and the
intervention metrics. README gives the commands. Complete decision history: `docs/05_decision_log.md`.

## References
* Atila, Sukhomlinov & Müser (2025). Cold self-lubrication of sliding ice. Phys. Rev. Lett. (arXiv 2402.10843). [landscape]
* Cordella, F. (2026). Odd denominators in the Lonely Runner spectrum for six speeds. arXiv 2609.03444.
* de Haan, Jayaraman & Levine (2019). Causal confusion in imitation learning. NeurIPS. arXiv 1905.11979.
* Fan & Sun (2023/2026). Amending the Lonely Runner Spectrum Conjecture. arXiv 2306.10417; Electron. J. Combin.
* Giri & Kravitz (2023). The structure of Lonely Runner spectra. arXiv 2304.01462.
* Hermann & Lampinen (2020). What shapes feature representations? Exploring datasets, architectures, and training. NeurIPS. arXiv 2006.12433.
* Hermann, Mobahi, Fel & Mozer (2023). On the foundations of shortcut learning. ICLR 2024. arXiv 2310.16228.
* Kawata, Song & Bietti (2025). From shortcut to induction head: how data diversity shapes algorithm selection in transformers. arXiv 2512.18634.
* Kravitz, N. (2021). Barely lonely runners and very lonely runners. Combinatorial Theory.
* Lanham et al. (2023). Measuring faithfulness in chain-of-thought reasoning. arXiv 2307.13702.
* Li, Liu, Zhou & Ma (2024). Chain of thought empowers transformers to solve inherently serial problems. ICLR. arXiv 2402.12875.
* Liu, Ash, Goel, Krishnamurthy & Zhang (2023). Transformers learn shortcuts to automata. ICLR. arXiv 2210.10749.
* Merrill & Sabharwal (2024). The expressive power of transformers with chain of thought. ICLR. arXiv 2310.07923.
* Sarnataro, Velasco, Monaco, Kempf & Miesenböck (2025). Mitochondrial origins of the pressure to sleep. Nature 645:722. [landscape]
* Shah, Tamuly, Raghunathan, Jain & Netrapalli (2020). The pitfalls of simplicity bias in neural networks. NeurIPS. arXiv 2006.07710.
* Shih, Winnicki & Darve (2026). When does activation steering change what a model computes from? arXiv 2606.29522.
* Valmeekam, Palod, Stechly, Gundawar & Kambhampati (2026). Beyond semantics: the unreasonable effectiveness of reasonless intermediate tokens. TMLR. arXiv 2505.13775.
* Wen, Lin, Darrell, Jayaraman & Gao (2020). Fighting copycat agents in behavioral cloning from observation histories. NeurIPS. arXiv 2010.14876.
* Yi, Gill & Bu (2026). Four runners that are barely lonely. arXiv 2610.40694.
* Zhong, Liu, Tegmark & Andreas (2023). The clock and the pizza: two stories in mechanistic explanation of neural networks. NeurIPS. arXiv 2306.17844.
* Zhu, Wang & Sui (2025). Chain-of-thought tokens are computer program variables. arXiv 2505.04955.
* Wang, Alazali & Zhong (2026). How does unfaithful reasoning emerge from autoregressive training? A study of synthetic experiments. arXiv 2602.01017.
* He, Fan, Xiao, Marculescu & Ribeiro (2026). Ceiling of a task: when can a transformer succeed without its chain of thought? arXiv 2609.33134.
* Garcia, G. (2026). The last word often wins: a format confound in chain-of-thought corruption studies. arXiv 2605.10799.
