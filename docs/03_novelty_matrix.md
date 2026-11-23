# Prior-art and novelty audit

Search log: `docs/search_log.md`. Full texts of the closest works were read (stored as text under `lit/`, not redistributed in git
if licence forbids; arXiv IDs given so anyone can retrieve them).

## A. Read vs recompute of written chain-of-thought intermediates

| # | Work | Setting | Intervention | Finding relevant to us | What it does NOT establish |
|---|---|---|---|---|---|
| 1 | "Ceiling of a Task" (arXiv 2609.33134, Sep 2026) | Group word problems A5, S5, Z60, A4×Z5; small transformers from scratch + open reasoning LLMs | Erase / replace whole chain by filler; shuffles | Above-ceiling tasks: chain is provably necessary. Empirically, chain-trained models on *non-serial* Z60 also drop to baseline when chain erased | Single-slot counterfactual corruption; regime where recomputing is cheaper than reading; any dial of relative cost |
| 2 | "How does unfaithful reasoning emerge…" (arXiv 2602.01017, Feb 2026) | 2-step modular arithmetic expressions, N = 97/29 | Replace reasoning step e2, measure whether answer follows | With noise-free chains models are step-wise faithful; noise above a threshold induces skip-step; larger complexity gap e1→e2 reduces unfaithfulness ("simplicity bias") | Zero-noise regime where skip/recompute is *simpler* than reading; multi-step chains; position dependence |
| 3 | "Beyond Semantics" (arXiv 2505.13775, TMLR 2026) | A* traces, pathfinding | Train on swapped/corrupted traces | Corrupted-trace training still works → traces need not be semantically used | Per-step causal reliance in correct-trace models |
| 4 | "The Last Word Often Wins" (arXiv 2605.10799) | GSM8K/MATH, open LLMs | Step corruption; answer-line removal | Corruption sensitivity tracks explicit answer placement | Synthetic control of computational cost |
| 5 | "Mechanistic Evidence for Faithfulness Decay" (arXiv 2602.11201) | Pretrained LLMs | Corrupt steps | Reasoning horizon beyond which steps have little effect | Mechanism of selection; controlled task |
| 6 | "Final Checkpoints Are Not Enough" (arXiv 2607.06648) | Latent (continuous) reasoning, ProsQA, GSM8K | Counterfactual edits on latent states across training | Responsiveness can decline while accuracy rises | Explicit token chains; cost dial |
| 7 | Lanham et al. 2023 "Measuring faithfulness in CoT reasoning" | LLMs | Truncation, mistakes, paraphrase | Reliance varies by task and model size | Controlled task with known algorithms |
| 8 | Merrill & Sabharwal 2023/24; Feng et al. 2023; Li et al. 2024 ("CoT empowers transformers to solve inherently serial problems") | Theory | — | Expressivity with/without CoT | Which strategy SGD selects when both are expressible |
| 8b | Liu, Ash, Goel, Krishnamurthy & Zhang, "Transformers Learn Shortcuts to Automata" (ICLR 2023, arXiv 2210.10749), App. B.2.3 | Automata (Dyck, C2, etc.), interleaved scratchpad | None (OOD accuracy only) | *Asserts* "the model can simply ignore the scratchpad positions and find the same parallel shortcut"; recency-biased attention needed to force recurrent use | No causal measurement of scratchpad use; directly in tension with #1 and #2 — the tension our experiment addresses |
| 10 | Hermann & Lampinen 2020 (arXiv 2006.12433); Hermann et al. 2023 "On the Foundations of Shortcut Learning" (arXiv 2310.16228); Shah et al. 2020 "Pitfalls of simplicity bias" | Synthetic redundant-feature tasks | — | Among redundant predictive features, models prefer the more *available* (easily extracted) one | Not about written intermediates / CoT; general principle our result instantiates |
| 11 | de Haan, Jayaraman & Levine 2019 "Causal Confusion in Imitation Learning" (arXiv 1905.11979); Wen et al. 2020 "Fighting Copycat Agents" (arXiv 2010.14876) | Behavioural cloning with action histories | Environment interventions | Policies copy their own previous action (cheap, predictive) instead of computing from observations | Opposite direction is also possible; no transformers/CoT, no cost dial |
| 12 | Shih, Winnicki & Darve 2026 "When Does Activation Steering Change What a Model Computes From?" (arXiv 2606.29522) | Controlled state-tracking, trace supervision | Activation edits of the internal register | Trace supervision creates an editable internal register used by subsequent steps | Edits internal activations, not written tokens; no regime where recomputation is cheaper |
| 13 | Zhu, Wang & Sui 2025 "Chain-of-Thought Tokens are Computer Program Variables" (arXiv 2505.04955) | LLM multiplication / DP | Edit CoT values | Downstream tokens follow edited values; notes unintended shortcuts | No control of relative cost |
| 14 | Zhong, Liu & Tegmark 2023 "The Clock and the Pizza" (arXiv 2306.17844); Kawata, Song & Bietti 2025 (arXiv 2512.18634) | Modular addition; trigger-copy task | Mechanistic analysis | Small hyperparameter/data changes select qualitatively different algorithms, sometimes mixtures | Not about chain reliance; relevant to any "mosaic" claim |
| 15 | Yi, Gill & Bu 2026 "Four runners that are barely lonely" (arXiv 2610.40694) | Lonely runner n = 4, general n | Exact search to 512; structural theorem | Settles n = 4; effective bound for all n | (secondary study) |
| 9 | Transformers provably learn to internalize CoT (arXiv 2605.28600) | k-parity theory, ICoT curriculum | — | Internalization under explicit curriculum | Spontaneous selection under full chains |

**Claimed difference (narrowed after adversarial review).** The principle that training selects the more available of redundant
zero-loss solutions is known (#10, #11). What we add is its demonstration for *written chain-of-thought intermediates* in transformers:
our experiment fixes everything except one dial (window K) that moves the relative cost of the two
zero-loss strategies through the point where recomputation becomes cheaper than reading, with noise-free teacher-forced chains,
and measures per-slot causal reliance with single-token and coherent-counterfactual interventions, corroborated by attention
signatures. To our knowledge (searches in log) no published study has (i) a zero-noise condition in which skipping the chain is the
*simpler* function, or (ii) a graded cost dial for strategy selection in chain-trained transformers. Closest is #2, whose complexity
manipulation always keeps reading simpler; a "simplicity bias" account from #2 *predicts* our H1, so our experiment is also a test of
that account outside its original regime.

**Risk to novelty:** unpublished/ongoing work; work under different vocabulary (e.g., "shortcut", "copy vs compute", "induction vs
algorithmic"). Additional searches for those terms are recorded in the search log.

## B. Lonely-runner spectrum computations

| # | Work | Radii searched | Relevance |
|---|---|---|---|
| 1 | Kravitz 2021 (Loneliness Spectrum Conjecture) | n ≤ 3 proved | Original conjecture |
| 2 | Fan & Sun (arXiv 2306.10417; EJC 2026) | experiments for n ≤ 7 | Amended conjecture 1 ≤ k ≤ n; counterexample family n = 4 |
| 3 | Giri & Kravitz (arXiv 2304.01462); Jain & Kravitz | theory | Accumulation points, relative spectra |
| 4 | Cordella (arXiv 2609.03444, Sep 2026) | n=4 ≤ 200, n=5 ≤ 130, n=6 ≤ 110, n=7 ≤ 50, n=8 ≤ 40 | Structure for n = 6, Questions 6.5–6.7 |

**Claimed difference.** Exhaustive searches at radii far beyond (4) for n = 7, 8 (and n = 6 for Q6.6), with exact arithmetic and an
independent implementation that reproduces every count of (4) on its radii. This is a data/verification contribution, not a theorem.
