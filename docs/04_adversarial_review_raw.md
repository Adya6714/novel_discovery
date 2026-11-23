# Adversarial review (raw) of the CoT read-vs-recompute study and the lonely-runner searches

Reviewer: a separate AI agent instance, asked to act as a skeptical outside referee. I did not design the study. This is
not human peer review. Date: 2026-11-23. State reviewed: branch `research/discovery-programme` at commit 69a5a9a plus
uncommitted logs; only seed-0 depth-2 checkpoints K = 1, 2, 3, 4 existed in `cot_reliance/runs/sweep/` at review time.

Scratch scripts I wrote (not part of the repository):
`/tmp/claude-0/-home-user/94c5e846-5cf7-5c8e-8992-50c6bc8cc22c/scratchpad/reviewer/{repro.py, attn_slot.py, v1.py, lr_check.py}`.

## 0. What I checked independently, and what held up

* **Reproduction (passes).** `repro.py` rebuilds the interleaved data, the chain, the index arithmetic and the single-token
  intervention from scratch. It uses only `model.GPT` and the checkpoints, with a different RNG stream. For K3_L2_s0 it gives FOLLOW = 1.000 at
  j = 2..5, 7..11, FOLLOW = 0.817 / RECOMPUTE = 0.036 at j = 1, and **FOLLOW = 0.000 / RECOMPUTE = 1.000 at j = 6**, which matches
  `results/sweep_results.json` to within sampling error. The K2 numbers (RECOMPUTE ≈ 1.00 everywhere) and the K4 numbers (FOLLOW = 1.00 at j ≥ 2,
  0.80 at j = 1) also reproduce. Clean teacher-forced accuracy is 1.000 at every slot for K = 2, 3, 4.
* **Index and formula audit (no bug found in the main path).** `InterWindowTask.slot_index(j) = 2j−1`, `pred_index(j+1) = 2j`
  (the position of x_{j+1}), `x_index(i) = 2(i−1)`. `follow_token` computes r + x_{j+1} − x_{j+1−K} correctly with 0-indexed x.
  The single-corruption r = c_j + U{1..10} is uniform over r ≠ c_j. The free-run recursion is correct. The gate is computed
  with an independent generator. `test_targets.py` passes logically for the interleaved layout.
* **Complementary (input-side) interventions, which the study does NOT run, confirm a double dissociation.** In
  `repro.py` I corrupted (b) the most recent interior input x_j, which only RECOMPUTE uses, and separately (c) the leaving
  input x_{j+1−K}, which only READ uses, while keeping the written chain fixed. At every j > K the result agrees with the c_j
  corruption. For K = 3: slot 6 tracks the x_j edit (1.000) and ignores the leaving-element edit (1.000 unchanged), while
  slots 4, 5, 7..11 do the opposite (1.000). For K = 2, all slots track x_j (1.000). So the behavioural classification at
  j > K is robust. **The objections below are about interpretation, confounds, novelty and generality, not about the main
  numbers being wrong.**
* **Lonely-runner kernel (passes).** I wrote an independent exact ML routine and compared it with `neartight` on all
  tuples for n = 4, V = 40 (174 = 174 primitive near-tight, identical sets) and n = 5, V = 26 (20 = 20, identical). Recompiling
  `neartight.c` reproduces the shipped binary's output for (n, V) = (6, 30) and (7, 28). Leaf counts in the chunk logs sum
  exactly to C(100,7), C(70,8) and C(55,9). ML(2,7,9,11,13,20,54) = 9/65 is confirmed by an exact routine and by a float grid
  (max 0.13842 < 9/65 = 0.13846).

---

## 1. Novelty and prior art

**1.1 (major) The general principle, "among redundant zero-loss solutions, training picks the cheaper/more available
one", is long established. Only its application to written CoT intermediates is new.**
- Hermann & Lampinen 2020, *What shapes feature representations?* (arXiv 2006.12433): when two features redundantly predict the
  label, the model prefers the more easily extractable one, and "an easier, weakly predictive feature can suppress a more
  strongly predictive, but more difficult one".
- Hermann et al., *On the Foundations of Shortcut Learning* (arXiv 2310.16228, ICLR 2024): availability versus predictivity in a synthetic two-feature
  framework, with an NTK account.
- Shah et al. 2020, *Pitfalls of simplicity bias*.
- de Haan, Jayaraman & Levine 2019, *Causal Confusion in Imitation Learning* (arXiv 1905.11979), and Wen et al. 2020, *Fighting Copycat
  Agents* (arXiv 2010.14876). These are the closest analogue of "read the previous written output versus recompute from the
  observation". Behaviour-cloned policies with access to their own past actions learn to copy them, because the
  previous action is a cheap, highly predictive feature. The interleaved window task is literally an autoregressive
  sequence-labelling problem with teacher forcing, which is the copycat setting.
- 2602.01017 (already cited) explicitly attributes faithful-vs-skip-step behaviour to simplicity bias. The novelty matrix
  admits that this account *predicts* H1.

So the headline claim, "selection governed by relative computational cost", is a corollary of simplicity bias,
availability and causal-confusion theory. The novelty matrix omits that whole literature. The "copy vs compute" and "shortcut"
risks noted in the matrix are real, and they were not resolved by the searches in `docs/search_log.md`.

**1.2 (major) The five closest works, and what is genuinely new relative to them.**

| Work | Overlap | What this study adds |
|---|---|---|
| Shih, Winnicki & Darve, *Do Models Read What They Write? Causal Registers in Scratchpad Reasoning* (arXiv 2606.29522, Jun 2026), **not cited** | Same question ("do later steps compute from the written state or recompute from the input?") on a controlled state-tracking task. Running-state supervision makes the written state causally used (80–91 %). Includes a "computed, not copied" control | A regime where recomputation is cheaper, and a graded dial |
| 2602.01017, *How does unfaithful reasoning emerge…* | Synthetic modular arithmetic. Step replacement. Simplicity bias decides faithful vs skip-step | Zero-noise regime where skipping is cheaper. Multi-slot position effects |
| 2609.33134, *Ceiling of a Task* | Chain-trained small transformers on Z60 (the abelian prefix-sum analogue of this study's K = 12) depend on the chain | The K = 12 control here **replicates known results**. New content is confined to K = 2..4 |
| Hermann & Lampinen 2020 / Hermann et al. 2023 | Availability decides between redundant features | Instantiation with CoT tokens as one of the features |
| de Haan et al. 2019 / Wen et al. 2020 (copycat) | A model reads its own previous output instead of computing from the observation when that is cheaper | Transformer/CoT framing; exact causal interventions |

Also relevant and uncited: *Chain-of-Thought Tokens are Computer Program Variables* (arXiv 2505.04955). It intervenes on CoT
values in multiplication/DP, finds that downstream tokens follow, and explicitly notes "unintended shortcuts" between tokens. Also
Zhong et al. 2023, *The Clock and the Pizza* (arXiv 2306.17844). It shows that small transformers on modular arithmetic select among
qualitatively different algorithms depending on hyperparameters, and run *mixtures* of them. The "mosaic" idea is a
position-wise version of that algorithmic-phase phenomenon and must be positioned against it. Also *From Shortcut to Induction
Head* (arXiv 2512.18634): data statistics select between a positional shortcut and an induction-head algorithm, with a
provable transition.

**What survives as new.** One thing survives: a demonstration, with exact behavioural interventions, that a small transformer trained
on correct, noise-free written intermediates ignores them entirely when recomputation from the input is cheaper (K = 2), and
uses them when it is not (K ≥ 3/4), on a task family with a single dial. That is a clean, citable toy result. It is an
incremental empirical point, not a new principle.

**1.3 (minor) The claim in REPORT §1, "In every prior controlled study we found, reading was the cheaper strategy", is
not accurate.** The copycat and causal-confusion literature has studied exactly the reverse trade-off. Liu et al. 2022 (cited) argued that a parallel
shortcut can be preferred to the scratchpad. 2602.01017's high-noise regime makes skipping the effectively cheaper option.

## 2. Methodology and code

**2.1 (critical, for the "mosaic" claim) The slot-6 recompute at K = 3 is very plausibly an absolute-position artefact. The
model learns position-specific circuits, and the analysis averages them away.**
The model uses learned **absolute** positional embeddings (`model.py`), with every training sequence of fixed length 12
(D15), so every slot is an identifiable absolute position. D14 justifies the interleaved layout by saying "every operand sits at a fixed relative offset". That does not
make the network translation-equivariant. It has to learn offset-attention separately at each absolute position, and nothing
forces the same circuit at each slot. My per-head attention analysis (`attn_slot.py`, clean sequences, n = 500) shows
this directly:
- K = 3, slots j = 4, 5, 7, 8, 9, 10, 11: a layer-0 head attends to offset −6 (the leaving element) with weight ≈ 0.8, and other heads attend to −1 (c_j).
- K = 3, slot j = 6 (query index 12): **no head attends to −6 or −1. Three of the four layer-0 heads attend to −2 (x_6), and layer 1 attends to 0.** That is a different, recompute-type circuit, gated by absolute position.
- K = 4, slot 6: all four layer-0 heads have their top key at −8 (leaving element) and none at −1, unlike slots 5 and 7. The circuits differ per slot even where behaviour does not.

So the "mosaic" is what one would expect from a network with 11 effectively separate per-position sub-problems and random
initialisation. Nothing ties it to "cost parity". `analyse_sweep.py` averages attention over all slots j ≥ K
(`r['attn']`), which hides exactly this heterogeneity. The averaged attention numbers in `sweep_results.json` should not be
used as mechanistic corroboration of a per-slot claim.

**2.2 (major) At j = K the "leaving element" is read from the chain token c_1, not from x_1, because c_1 ≡ x_1 in every
training sequence. This is a training-data redundancy that contaminates the j ≥ K averages.**
- Corrupting the input x_1 at j = K: K = 3 follows the edit only 36 % of the time, and K = 4 only **1.6 %** (85.5 % unchanged). In contrast, at j > K the model follows the leaving-element edit 100 % of the time (`repro.py`).
- Attention at j = K points at c_1: offset −5 for K = 3 (index 1 from query 6), and −7 for K = 4.
- Under the consistent corruption, K = 4 at j = 4 gives FOLLOW 0.10 / RECOMPUTE 0.09, so about 81 % OTHER. K = 3 at j = 3 gives FOLLOW 0.43.

`check_jK.py` exists, so the authors noticed this. But the preregistered primary outcome, "mean FOLLOW over j ≥ K", still
includes j = K. There are more redundancies of the same kind: for j < K, c_j is a prefix sum, so x_i = c_i − c_{i−1}. The j = 1 deficit
(FOLLOW ≈ 0.80–0.82 at K = 3, 4) comes from the same c_1 = x_1 duplication. **Fix:** break the identities, for example by
prepending a random, visible offset c_0 to the chain or by starting the window with K−1 burn-in inputs that are never
chain-labelled. Alternatively, report j = K and j < K separately and exclude them from the cost claim.

**2.3 (major) The docstring claim for the consistent corruption ("Reading *any* chain token then implies FOLLOW";
REPORT §5: "A model that reads any chain token follows; only input-recomputation ignores it") is false.**
A model that reads an older chain token c'_{j−1} and combines it with *visible* inputs predicts c'_{j−1} + x_j − x_{j−K} +
x_{j+1} − x_{j+1−K}. That differs from FOLLOW (c'_j + x_{j+1} − x_{j+1−K}), because x'_j ≠ x_j in the counterfactual. Such reading lands in
OTHER, as the j = K data show (item 2.2). So the consistent intervention rules out "older chain token" reading only for models that
score RECOMPUTE ≈ 1 under it. That covers K = 2 and K = 3 slot 6, so the conclusion survives there. But the general statement in the
methods is wrong and should be corrected. OTHER needs its own decomposition into named alternative strategies.

**2.4 (major) REPORT.md §4–5 describe the abandoned design, not the one that produced the results.** REPORT says:
- layout `x_1..x_L, SEP, c_1..c_L` (actual: interleaved `x1 c1 x2 c2 …`);
- L ~ U{2..12} (actual: fixed L = 12, D15);
- 4000 steps, continuation blocks of 6000 up to 22,000 (actual: 3000 / 3000 / cap 18,000, D16);
- "the model's input token is c_j; it predicts c_{j+1}" (actual: the input token is x_{j+1}).

The operand table is also wrong for the interleaved layout. x_{j+1} is the query token itself, so the operands that must be
*retrieved by attention* are 2 (READ: c_j at −1, x_{j+1−K} at −2K) versus K−1 (RECOMPUTE). This still gives parity at K = 3, but the document as written
is internally inconsistent with the code.

**2.5 (minor) Bug: the untrained negative control uses the wrong task layout.** `analyse_sweep.py --untrained` builds
`make_task(f'win:11:{K}')`, which is the abandoned v1 `WindowTask`, with `vocab = p+1 = 12`, instead of `iwin`. The control therefore
does not match the models it controls for.

**2.6 (minor) "Validated the intervention metric on exact READ / RECOMPUTE oracle models (FOLLOW = 1.00 / RECOMPUTE = 1.00)"
(D14) has no code or output anywhere in the repository** (`grep -ri oracle` finds only the decision-log sentence). It cannot be
verified and should be committed or withdrawn.

**2.7 (minor) Argmax classification under an off-distribution input measures *dominance*, not *exclusive use*.** Training chains
are always consistent, so a corrupted c_j is out of distribution. A model that computes both READ and RECOMPUTE and sums their
logits would be classified by whichever pathway has the larger margin. The K = 3/4 per-slot results are near 0/1, which argues
against heavy mixing, but `p_follow` / `p_recompute` should be reported per slot, and an activation-level test (path patching
of the c_j → query head) is needed before any claim that a slot "does not use" the chain.

**2.8 (minor) The gate is weak as a gate.** All models reach 1.000 teacher-forced accuracy by step 2000 (logs), so the gate never binds
at depth 2. Training stops at 3000 steps, about 1000 steps after reaching zero error, with lr annealed by the one-cycle schedule. Strategy
selection under weight decay can drift long after the loss saturates (grokking-style circuit cleanup). The slot-6 choice may not be
stable, and is not shown to be.

## 3. Alternative explanations and confounds

**3.1 (critical) "Computational cost" is not operationalised independently of the outcome, so the main claim is close
to unfalsifiable as stated.** The "cost" used is operand count. In a transformer:
- Attention gathers several tokens in parallel. With 4 heads per layer, both strategies fit in one attention layer for every K ≤ 5. Depth cost is equal.
- At K = 3, READ (a + b − c over two token types) and RECOMPUTE (a + b + c over one token type) are the *same function class* (a 3-term signed sum mod 11). RECOMPUTE even gets weight sharing (one embedding table, no sign flip). By that argument recompute should be favoured at "parity", yet 10 of 11 slots READ.
- Other plausible cost measures give different crossovers. An MLP lookup over ordered operands gives 11^3 vs 11^K, crossover at K = 3. A lookup over *multisets* (RECOMPUTE is order-free) gives 1331 ordered READ inputs vs C(K+10, 10) = 286 (K = 3), 1001 (K = 4), 3003 (K = 5) RECOMPUTE multisets. That predicts recompute at K = 3 **and** K = 4, contrary to the data.

With only one transition (between K = 2 and K = 3), any monotone cost proxy fits. The claim needs a *pre-specified,
independently measured* cost. For example, train READ-only and RECOMPUTE-only models, enforced by attention masks, and measure
steps-to-criterion or minimum width. Then test whether the free model picks the strategy that is learned faster, slot by slot.

**3.2 (major) K is confounded with several things other than relative cost.**
- (a) The offset of the leaving element (−2K) grows with K, while the RECOMPUTE operands stay within −2(K−1).
- (b) The number of distinct positions the strategy needs.
- (c) The proportion of slots j < K and the prefix-sum redundancies of item 2.2.
- (d) Token-type: READ always uses the *adjacent* token (−1), which is a separate vocabulary that the model also learns to emit.
  An "adjacent-token / most-recent-output preference" (copycat) explains K ≥ 3 READ, including parity, without any cost
  argument. K = 2 is then the only case where the input route is strictly simpler.
- (e) For K = 1 the "positive control" is a copy of the current token (zero retrieved operands). It validates the metric, not the theory.
- (f) The v1 failure (D14) shows that READ learnability depended on the *layout*: in v1, K ≥ 3 models never learned the subtraction (my `v1.py`: K3/K4 v1 at chance beyond slot 2), while K = 2 recomputed. The design was changed until READ became learnable. "Cost" therefore includes layout- and positional-encoding-dependent learnability that the theory does not model.

**3.3 (major) Is RECOMPUTE at K = 2 reading an older chain token?** No, as far as I can test. RECOMPUTE ≈ 1.00 under the consistent
counterfactual, and my x_j-edit test shows the prediction tracks the visible input x_j 100 % of the time. K = 2 is the
most solid result. It is also the least surprising: c_{j+1} = x_j + x_{j+1} is a two-token bigram sum at fixed offsets 0 and −2.

**3.4 (major) The K = 3 data argue against "mosaics at parity" as a general phenomenon.** At the parity slots (K = 3: j ≥ 3 and j = 1;
K = 4: j = 1) the models READ in about 10 of 11 cases. The single exception has the position-specific circuit described in item 2.1. Read
neutrally, the evidence currently supports a *read preference at parity*, with one idiosyncratic slot in one seed.

**3.5 (minor) Relevance to CoT faithfulness is overstated.** With constant K the whole task is computable in one forward pass at any
length. The chain never bridges serial depth, so this is an autoregressive multi-output labelling task (exposure-bias/copycat
regime), not CoT in the sense of 2609.33134. Generalising to "CoT traces as a window into model computation for AI safety"
(REPORT §1) needs that caveat.

## 4. Statistics and claims

**4.1 (critical) The "mosaic" claim rests on n = 1 slot in n = 1 seed.** n = 2000 sequences per slot gives SE ≤ 0.011 for *that
model's* behaviour. It is pseudo-replication with respect to the scientific claim, whose unit is the trained model, or the (model, slot)
pair. With 3 seeds, if a mosaic appears in 1 of 3, the 95 % Clopper–Pearson interval for its probability is [0.008, 0.906]. The
experiment cannot distinguish "mosaics are typical at parity" from "rare idiosyncrasy". Characterising per-slot strategy
distributions needs on the order of 20+ seeds at K = 3, plus variation in width, lr, wd and init scale (a Clock/Pizza-style
phase diagram). Each seed costs about 12 CPU-minutes, so this is affordable.

**4.2 (major) What 3 seeds *can* support:** the K = 2 (all RECOMPUTE) versus K ≥ 4 (all FOLLOW) contrast, if all seeds agree. The effect is
0 versus 1, and 3/3 agreement in each cell is convincing for that coarse claim. They cannot support: a quantitative K-curve, a
claim about the exact crossover location, any per-slot pattern, H3 (depth), or any claim of generality beyond d = 128 / 4 heads /
absolute PEs / p = 11 / L = 12.

**4.3 (major) Preregistration status is weaker than presented.**
- The preregistration specifies the v1 layout, variable L, depths 2 and 4, and "steps fixed per cell".
- The claims will come from a different layout (D14), fixed L (D15) and a different budget (D16). These were chosen after window-task pilots (`wpilot*`, `ipilot*`) had been trained and inspected. D15 records pilot observations ("K=2 RECOMPUTE ≈1.00; K=12 FOLLOW ≈1.00; K=4 FOLLOW = 1.00") *before* the sweep.
- The "frozen" status rests on a local file mtime (06:40). The first commit is at 07:21, after the `wpilot` runs (06:49–07:01). There is no external registry.
- The "mosaic at cost parity" hypothesis appears nowhere in the preregistration. It is post hoc and must be labelled exploratory.
- H2 (lower FOLLOW at j < K) is essentially unsupported by the K = 3/4 data (FOLLOW ≈ 0.8–1.0 at j < K). The j = 1 dip is the c_1 = x_1 artefact, and should be reported as such, not as support.

**4.4 (minor) Primary outcome contamination.** The preregistered "mean FOLLOW over j ≥ K" mixes j = K (chain-leave artefact,
item 2.2) with the clean slots. Report per-slot results, with j = K separated.

## 5. Lonely-runner secondary study

**5.1 (passes) Exactness argument.**
- G = ∩G_i is closed and symmetric under t → 1−t, and 0 ∉ G, so every component lies in (0,1). A component's left endpoint is a left endpoint of some G_i interval.
- A component with left endpoint > 1/2 mirrors to one whose left endpoint 1 − r is < 1/2. So testing left endpoints (na+1)/(n v_i) ≤ 1/2 suffices, and degenerate (point) components are handled by the inclusive test.
- The membership test r ∈ [v_i, (n−1)v_i] for r = (na+1)v_j mod n v_i is correct.
- Overflow is impossible: num·v_j < 10^5 in int64.
- MAXP = 200,000 far exceeds the pool size (at most about nV/2 ≈ 500).
- The primitive filter is harmless, because ML is invariant under scaling and every imprimitive tuple is a dilate of a primitive one with smaller speeds.
- `ml_exact.py`'s candidate set t = m/(v_i+v_j), with i = j allowed, is the standard complete set of local maxima of a min of tent functions.
- My independent cross-check agrees exactly (section 0).

**5.2 (major) Significance is low, and the novelty matrix misses the most relevant recent work.**
- (a) **Yi, Gill & Bu, *Four runners that are barely lonely* (arXiv 2610.40694, 28 Oct 2026), not cited.** They settle n = 4 completely, with an exact search over 8×10^8 quadruples, and, conditionally for every n, prove that every near-tight n-tuple with |v| ≥ n(n(n−1))^{2n−4} lies on a 2-D subtorus of loneliness 1/n. That answers Cordella's Question 6.7 in effective form. Bounded radius searches at n = 7–9 are now much less informative than that structural result.
- (b) Cordella's code is public (Zenodo) and "reproduces every number in about twenty minutes on two cores". Extending radii from 50/40 to 100/70/55 is a direct scaling of an existing computation that any of these authors could run. The "new" k = 2 septuple and the absence of k ≥ 3 up to the radius are data points, not structure.
- (c) Finding 4 (tight septuples (1..7), (1,2,3,4,5,7,12), (1,4,5,6,7,11,13)) is a rediscovery of known tight instances: Goddyn–Wong type, and the sporadic sets in the tight-instance literature, e.g. arXiv 2608.13599. It needs citation, not a "finding" label.
- (d) Most n = 8, 9 hits are the trivial families (1,…,n−1, n·s) (`nt_n9_V55.txt`: (1..8, 18), (1..8, 27), …). The only conjecture supported by the data, the amended Fan–Sun conjecture, is supported only at small radius.
- Realistic outlet: a data appendix to, or contribution towards, the Cordella / Yi–Gill–Bu programme. Not a standalone paper.

**5.3 (minor) The leaf-count check verifies enumeration completeness, not correctness of the pruning.** The independent
small-radius cross-check (mine) is the meaningful validation and should be added to the repository.

## 6. Most informative falsification experiments (in priority order)

1. **Independent cost manipulation at fixed K (tests the core claim).** At K = 3, make READ costlier without changing operand count. Options: write the chain through a fixed random permutation of symbols (reading then needs a lookup), multiply the written value by a hidden constant, or insert filler so c_j sits farther away. Alternatively make RECOMPUTE costlier (inputs written as two-token codes). Separately, *measure* cost: train attention-masked READ-only and RECOMPUTE-only models and record steps-to-criterion per slot. Cost theory predicts that the selected strategy flips with the manipulation and matches the faster-learned strategy per slot. If READ persists when it is clearly costlier, the claim is falsified in favour of an adjacency/copycat preference.
2. **Positional-encoding control (tests whether mosaics are an artefact).** Retrain K = 3 (≥ 10 seeds) with RoPE or ALiBi, or with absolute PEs plus random global position offsets (maxlen ≫ 24, random start). Also try variable L in the interleaved layout. If mosaics vanish or become random across seeds, the "mosaic" claim is an absolute-position artefact.
3. **Seed/hyperparameter census at K = 3 and K = 2.5-like intermediates.** Run 20+ seeds, plus width/wd/lr sweeps, recording per-slot strategy. Construct intermediate costs, e.g. a weighted window c_j = x_j + 2x_{j−1} + x_{j−2}, or K = 3 with one operand duplicated in the chain. Estimate the probability of RECOMPUTE per (K, slot) with CIs. Look for the Clock/Pizza-style phase structure.
4. **Training-dynamics and long-training stability.** Save checkpoints every 250 steps and continue to 30k steps with wd ∈ {0.01, 0.1, 1.0}. Track per-slot strategy and parameter norm. If slot 6 flips under longer training or higher wd, the "choice" is a transient of optimisation, and cost should be defined in terms of norm or regularisation.
5. **Remove training-data redundancies and add input-side interventions to the pipeline.** Break c_1 = x_1 and the prefix-sum identities (random visible c_0 offset, or burn-in inputs). Add the input-corruption double dissociation used in this review, plus path patching of the c_j → query head, as standard outputs.

## 7. Publication significance

- **As is (seed 0, K ≤ 4, one anomalous slot):** not publishable as a claim about strategy selection. The robust content, K = 2 recompute versus K ≥ 4 read, would make a short workshop paper. Candidates are a NeurIPS/ICLR workshop on mechanistic interpretability, reasoning or faithfulness, or a BlackboxNLP-style short paper.
- **With 3 seeds per K and a corrected report:** still workshop tier, because the principle is known (item 1.1).
- **With experiments 1–3 done well, especially an independently measured cost that predicts selection, and with the copycat/availability literature positioned properly:** a TMLR paper, or a borderline main-track ICLR/NeurIPS submission. The mosaic claim would need to survive the PE control to be a selling point.
- **Lonely runner:** a data note or supplementary material. Not a standalone publication in a good venue after arXiv 2610.40694.

## 8. Summary of severities

| # | Objection | Severity |
|---|---|---|
| 2.1 | Slot-6 "mosaic" coincides with a position-specific circuit under absolute PEs; averaged attention hides it | critical (for mosaic claim) |
| 3.1 | "Cost" not operationalised independently; alternative cost proxies predict different crossovers | critical |
| 4.1 | Mosaic claim = 1 slot × 1 seed; pseudo-replication | critical |
| 1.1 | Principle already known (simplicity bias, availability, causal confusion/copycat) | major |
| 1.2 | Closest works missing: 2606.29522, 2505.04955, Clock & Pizza, 2512.18634, copycat literature | major |
| 2.2 | j = K reads x_1 from c_1 (data redundancy); contaminates the primary outcome | major |
| 2.3 | Consistent-corruption claim "reading any chain token ⇒ FOLLOW" is false | major |
| 2.4 | REPORT methods describe the abandoned design | major |
| 3.2 | K confounded with offset, adjacency/token-type, layout learnability | major |
| 3.4 | Parity data favour read-preference, not mosaics | major |
| 4.2–4.3 | 3 seeds support only coarse contrast; preregistration compromised by post-pilot redesign; mosaic hypothesis post hoc; H2 unsupported | major |
| 5.2 | Lonely runner: low significance; misses arXiv 2610.40694; rediscovered tight sets | major |
| 2.5–2.8, 1.3, 3.5, 4.4, 5.3 | Untrained control wrong layout; oracle validation absent; argmax = dominance; non-binding gate/short training; overstated framing | minor |
