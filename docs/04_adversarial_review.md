# Adversarial review and responses

The full raw review, written by a separate AI agent instance that did not design the study, is in
`docs/04_adversarial_review_raw.md`. It is **not** human peer review and **not** independent experimental replication.
The reviewer did, however, independently re-implement the interventions and reproduce our numbers (K=3 seed 0 slot 6 RECOMPUTE = 1.000,
etc.) and independently cross-checked the lonely-runner kernel. Below, every objection is answered: **[accepted]** = we changed
claims/code; **[tested]** = we ran a new experiment (results in REPORT §7); **[disputed]** = we disagree, with reasons.

Reference check on the reviewer's citations: all arXiv IDs exist (verified via the arXiv API). One title was misquoted by the
reviewer: arXiv 2606.29522 is "When Does Activation Steering Change What a Model Computes From?" (Shih, Winnicki, Darve); it shows that
trace supervision creates an internally editable register in a state-tracking task (interventions on activations, not written tokens).

| # | Objection (severity) | Response |
|---|---|---|
| 2.1 | Slot-6 "mosaic" may be an absolute-position artefact (critical) | **[tested]** Retrained with random absolute-position offsets (0..16) per batch, K=3 × 3 seeds (+K=2, K=4), evaluated at offsets 0/8/16; per-slot census. **[accepted]** attention is now reported per slot, not averaged, for any per-slot claim. See REPORT §7.2 |
| 3.1 | "Cost" not operationalised independently (critical) | **[tested]** Independent cost = learnability of each strategy in isolation: forced-READ and forced-RECOMPUTE models enforced by attention whitelists, steps-to-criterion and AUC, K ∈ {2,3,4,6} × 2 seeds. The cost claim is now stated *only* in that operational form. See REPORT §7.1 |
| 4.1 | Mosaic = 1 slot × 1 seed (critical) | **[tested]** 6 additional K=3 seeds (9 total) with per-slot census; Clopper–Pearson intervals reported. **[accepted]** mosaic is labelled exploratory/post hoc |
| 1.1 | Principle known (simplicity bias, availability, copycat) (major) | **[accepted]** Novelty matrix and REPORT now position the work as an instance of availability/simplicity bias (Hermann & Lampinen 2020; Hermann et al. 2023; Shah et al. 2020) and the copycat problem (de Haan et al. 2019; Wen et al. 2020), applied to written CoT intermediates. Claimed novelty narrowed to: a single-dial demonstration, with exact causal interventions and an independently measured cost, that chain-trained transformers *ignore correct written intermediates* when recomputation is cheaper, and the specific crossover |
| 1.2 | Missing closest works (major) | **[accepted]** Added 2606.29522, 2505.04955, Clock & Pizza (2306.17844), 2512.18634, Hermann et al., copycat papers |
| 1.3 | "Every prior study had reading cheaper" inaccurate (minor) | **[accepted]** Sentence removed; copycat literature studies the reverse trade-off in imitation learning |
| 2.2 | j = K reads x_1 from c_1 (major) | **[accepted]** We had found the same (check_jK.py: K=4 85% chain-leave). Primary outcome now reported over j > K ("jgtK"), with j = K and j < K reported separately; the j = 1 dip is attributed to the c_1 = x_1 redundancy, not to H2 |
| 2.3 | "Reading any chain token ⇒ FOLLOW" false (major) | **[accepted]** Statement corrected: the coherent counterfactual rules out older-chain reading only for models that score RECOMPUTE ≈ 1 under it (K=2, K=3 slot 6), which is the case in which we use it |
| 2.4 | REPORT described abandoned design; operand table wrong for interleaved layout (major) | **[accepted]** Methods rewritten for the interleaved layout, fixed L=12, 3000 steps; operand table now counts *retrieved* operands (READ: c_j at −1 and x_{j+1−K} at −2K = 2; RECOMPUTE: K−1) |
| 3.2 | K confounded with leaving-element offset, adjacency/copycat preference, layout learnability (major) | **[partly accepted]** The forced-learnability experiment measures cost *including* these factors, which is the point: we no longer claim operand count is the cost. Adjacency preference is an alternative we cannot exclude beyond what §7.1 shows; stated as a limitation |
| 3.3 | Is K=2 RECOMPUTE older-chain reading? (major) | Reviewer's own test answers no (input x_j edit tracked 100%); we added the same input-side intervention to the pipeline (`intervene_input`) |
| 3.4 | Parity data favour read-preference, not mosaics (major) | **[accepted]** as the primary reading; see census |
| 3.5 | Relevance to CoT safety overstated (minor) | **[accepted]** Added caveat: constant-K tasks are below the ceiling (the chain never bridges serial depth); results concern the *below-ceiling* regime only — which is exactly where corruption-based faithfulness tests are ambiguous |
| 4.2 | 3 seeds support only coarse contrast (major) | **[accepted]** Claims restricted accordingly |
| 4.3 | Preregistration compromised by post-pilot redesign; mosaic post hoc; H2 unsupported (major) | **[accepted]** REPORT states plainly that the layout (D14), training length (D15) and budget (D16) changed after pilots were inspected, that the preregistration timestamp is a local file time only (first commit 07:21 UTC), that the mosaic analysis is exploratory, and that H2 is not supported |
| 2.5 | Untrained control used wrong layout (minor) | **[accepted]** fixed (`iwin`) |
| 2.6 | Oracle validation not in repo (minor) | **[accepted]** `test_oracle.py` committed (also covers the input-side intervention) |
| 2.7 | Argmax = dominance, not exclusivity (minor) | **[accepted]** p_follow/p_recompute reported; language changed to "dominant route" |
| 2.8 | Gate never binds; short training (minor) | **[tested, partly]** training-dynamics run (checkpoints every 250 steps) for one K=3 seed; long-training (30k) not done — listed as future work |
| 5.2 | Lonely runner: low significance; misses Yi–Gill–Bu 2610.40694; tight sets known (major) | **[accepted]** Cited; tight septuples relabelled as known instances; significance downgraded to "data note" |
| 5.3 | Add independent small-radius cross-check (minor) | **[accepted]** see `lonely_runner/crosscheck.py` |
