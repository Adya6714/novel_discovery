# Part I–III: Research landscape, feasibility audit and prioritisation

Date: 2026-11-23. Author: autonomous AI research agent (Claude), see `docs/06_human_vs_ai_contribution.md`.

## 0. Binding constraint: what this research organisation can actually do

Before ranking anything we audited our real resources, because a "discovery opportunity" that cannot be
executed is not an opportunity for us.

| Resource | Available? |
|---|---|
| Wet lab, instruments, animals, human participants | **No** |
| Compute | 4 CPU cores, 15 GB RAM, no GPU, ~1 working session |
| Pretrained LLM weights (HuggingFace) | **No** (host blocked) |
| Literature | arXiv full text, OpenAlex, Crossref, Semantic Scholar, Europe PMC, OEIS (granted mid-session) |
| Software | PyPI (PyTorch CPU, NumPy, etc.), GitHub |

Consequence: programmes whose decisive next step is a physical measurement (1, 2, 3, 5, 7, 11, 13 and most of 4, 6, 12)
can be *audited* here but not *advanced* by original data. We say so explicitly instead of producing
paper-shaped speculation. Original research is feasible in (a) mathematics with rigorous computer verification,
(b) controlled machine-learning experiments on models we train ourselves, (c) re-analysis of downloadable public data.

## 1. Audit of the 13 supplied programmes

Status claims below were checked against the primary sources listed (search log: `docs/search_log.md`).

### P1 Ice friction
* **Established:** pressure melting is quantitatively insufficient at ordinary loads (classic Bowden–Hughes argument);
  premelted quasi-liquid layers exist at the free ice surface below 0 °C.
* **Open:** Atila, Sukhomlinov & Müser (PRL, Aug 2025, "Cold self-lubrication of sliding ice") find by MD that sliding
  amorphises ice and the amorphous layer grows with sliding distance even far below melting; premelting matters only
  initially. They explicitly conclude that "either the main reason for the presence of water ... varies from case to
  case, or some crucial ice-liquefaction mechanism has not been accounted for". A 2026 preprint (arXiv 2603.11539)
  on ice/amorphous-silica argues nanoscale MD cannot capture the experimental velocity dependence and supports frictional heating.
* **Precise remaining uncertainty:** crossover sliding distance/velocity/temperature at which amorphisation-dominated
  lubrication hands over to flash-heating; MD timescales are 10^6 too short to overlap tribometer data.
* **Research questions:** (i) Does a combined amorphisation-rate + heat-balance model reproduce the measured
  minimum of μ(T) near −7 °C? (ii) Does counter-surface hydrophobicity shift the crossover? (iii) Do multiscale models
  predict a sliding-distance dependence of μ at very low T that tribometers could detect?
* **Feasible here?** Only a phenomenological re-fit of published μ(T,v) curves; decisive tests need MD on GPUs or a
  cryo-tribometer. **Not selected.**

### P2 Sleep pressure and mitochondria
* **Established (causal, in flies):** Sarnataro, Velasco, Monaco, Kempf & Miesenböck, *Nature* 645:722 (2025): sleep
  loss up-regulates mitochondrial transcripts in dFB neurons; fission/fusion manipulations move sleep in opposite
  directions; reducing electron leak cuts sleep need; optogenetic ATP synthesis triggers sleep. Builds on Kempf et al.
  *Nature* 2019 (Hyperkinetic/Kvβ redox sensor).
* **Open:** whether the mechanism exists in mammals ("formally, it needs to be proven in mammals"); how it relates to
  adenosine/Homer1a/synaptic-phosphorylation (SNIPPs) accounts in mice; why only some populations act as sensors.
* **Feasible here?** Re-analysis of public mouse sleep-deprivation transcriptomes (GEO) for mitochondrial-ETC signatures
  by cell type is possible in principle, but GEO was not reachable and correlational re-analysis cannot reach the
  causal bar the programme sets. **Not selected.**

### P3 General anaesthesia
* **Established:** GABA_A β3-N265M knock-in mice are resistant to etomidate/propofol immobility; xenon/ketamine act via
  NMDA; network signatures (frontal alpha, loss of complexity/PCI) are agent-dependent.
* **Open:** the propagation from receptor to network; separation of unconsciousness from unresponsiveness (disconnected
  consciousness under ketamine/dexmedetomidine).
* **Feasible here?** Public EEG data (e.g., propofol datasets on OpenNeuro) could test transition predictors, but the host
  was not available and the field already has many such analyses. **Not selected.**

### P4 Aging biology
* **Established:** senolytics extend healthspan in mice; epigenetic clocks predict mortality; rapamycin is the most reproducible
  pharmacological intervention (ITP).
* **Open (from the 100 Open Problems list and roadmaps):** causal status of clock CpGs (damage readout vs driver); tissue
  specificity of hallmark ordering.
* **Feasible here?** Longitudinal biobank data needs controlled-access approval. **Not selected.**

### P5 Nanomaterial synthesis reproducibility
* **Established:** a 2026 systematic review (arXiv 2607.14849, Turkevich Au NPs and CVD MoS2) notes that few studies isolate
  individual parameters with statistically robust designs; interlab *measurement* studies (VAMAS, Nanoscale 2022) exist.
* **Open:** quantitative attribution of variance to unreported parameters (stir rate, vessel, water quality).
* **Feasible here?** A text-mining meta-analysis of reported Turkevich conditions vs size would be possible but would be
  limited by reporting gaps that are the very problem. Requires lab work to be decisive. **Not selected.**

### P6 Non-coding RNA
* Large literature, mostly associational; decisive work needs perturbation experiments. Public CLIP/RNA-seq
  re-analysis is possible but EBI/ENCODE throughput here is limited. **Not selected.**

### P7 Supramolecular cooperative assembly
* Established nucleation–elongation models (Goldstein–Stryer, Meijer/de Greef). Open questions concern pathway
  complexity and kinetic traps; decisive tests are spectroscopic. **Not selected.**

### P8 LLM mechanistic interpretability
* **Established:** CoT can be unfaithful in LLMs (Turpin et al. 2023; Lanham et al. 2023); corrupted-trace training can still
  work (Stechly/Kambhampati et al., "Beyond Semantics", TMLR 2026); in synthetic arithmetic, noisy training drives a transition
  from faithful step-wise to skip-step reasoning (arXiv 2602.01017, Feb 2026); for *serial* tasks (above the "ceiling" of a
  constant-depth circuit), the chain is provably necessary (arXiv 2609.33134, Sep 2026); corruption studies on GSM8K are
  confounded by answer placement (arXiv 2605.10799).
* **Precise remaining uncertainty (identified by us):** on tasks *below* the ceiling, correct-chain training is
  loss-indifferent between *reading* the written intermediate and *recomputing* it from the input. Which strategy gradient
  descent selects, and whether the choice depends on how hard the intermediate is to compute in one pass, is not settled by
  the necessity theorems (which constrain only above-ceiling tasks) nor by noise-driven studies.
* **Feasible here?** Yes — small transformers trained from scratch on CPU, with exact causal interventions. **Selected (primary).**

### P9 ML interatomic potentials
* **Established:** systematic PES softening of universal MLIPs (Deng et al., npj Comput Mater 2025), fixable by a linear
  correction from one DFT label.
* **Open:** whether uncertainty estimates flag softening; structural predictors of failure.
* **Feasible here?** Needs DFT references; small-molecule semi-empirical references would not test the real failure mode.
  **Deferred.**

### P10 Accessible mathematics
* **State of field (2026):** Erdős problems are being resolved at scale by AI systems (≈100 moved to "solved" since Oct 2025;
  DeepMind *Aletheia* study arXiv 2601.22401; OpenAI unit-distance counterexample May 2026). The Lonely Runner Conjecture is now
  computer-verified up to 15 runners (Rosenfeld 2025; Trakulthongchai 2025; Sungkawichai–Trakulthongchai 2026; arXiv 2609.02604).
  The LR spectrum for six speeds was largely resolved by Cordella (arXiv 2609.03444, Sep 2026).
* **Precise open items found:** Cordella Q6.6 (does max v_i < 3q hold for all near-tight sextuples?); amended Fan–Sun
  conjecture for n = 7 (near-tight septuples searched only up to speed 50; blocked theoretically by classification of tight sextuples).
* **Feasible here?** Yes (exact rational computation). Highly crowded; novelty requires going beyond existing search radii. **Selected (secondary).**

### P11 Animal culture, P12 Ecological patterns, P13 Placebo
* P12: 2025 review "Fairy circle research: status, controversies and the way forward" — termite vs self-organisation still disputed;
  a global PNAS analysis found termites unimportant except in Namibia. Satellite time-series analysis needs Earth-Engine access we lack.
* P11, P13: need field/participant data. **Not selected.**

## 2. Expansion beyond the agenda (additional candidates)

| ID | Candidate | Why it might matter | Feasible here |
|---|---|---|---|
| X1 | Error propagation vs self-correction in chain-trained transformers (does a corrupted step propagate?) | Directly bears on CoT monitoring for AI safety | Yes |
| X2 | Group-theoretic predictors of CoT reliance (solvable vs non-solvable) | Links circuit complexity to *learned* faithfulness | Yes |
| X3 | Tight lonely-runner sextuples beyond speed 130 | Would unblock n = 7 spectrum | Partly |
| X4 | Exact max-loneliness / denominator ratio constants C(n) | Cordella Q6.6 | Yes |
| X5 | Training-dynamics of read-vs-recompute (when is the strategy fixed?) | Mechanism of faithfulness emergence | Yes |
| X6 | Grokking-style delayed circuit switch in CoT models | Relates to latent-reasoning faithfulness drift (arXiv 2607.06648) | Yes |
| X7 | Re-analysis of public tribology μ(T,v) data for a universal crossover | P1 boundary condition | Partly |
| X8 | OEIS sequences of LR-spectrum values | Low impact | Yes |

## 3. Ranking (1–5 scale; I = importance, U = unresolved status, N = novelty, T = tractability *here*, R = resources, D = data,
S = surprise potential, Rep = reproducibility, P = publication potential, Sm = small-team suitability)

| Rank | Opportunity | I | U | N | T | R | D | S | Rep | P | Sm | Σ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P8/X1–X2: read vs recompute of written intermediates below the ceiling | 4 | 4 | 4 | 5 | 5 | 5 | 4 | 5 | 4 | 5 | 45 |
| 2 | X5: when during training is the read/recompute strategy fixed | 3 | 4 | 4 | 5 | 5 | 5 | 4 | 5 | 3 | 5 | 43 |
| 3 | P10/X4: Cordella Q6.6 constant for n = 6/7 | 3 | 5 | 3 | 4 | 4 | 5 | 3 | 5 | 3 | 5 | 40 |
| 4 | P10: amended Fan–Sun conjecture, n = 7 extended search | 3 | 5 | 3 | 3 | 3 | 5 | 3 | 5 | 3 | 5 | 38 |
| 5 | X6: delayed circuit switch in CoT models | 3 | 4 | 4 | 4 | 4 | 5 | 4 | 4 | 3 | 5 | 40 |
| 6 | P2: mammalian test of mitochondrial sleep-pressure signal (transcriptome re-analysis) | 5 | 5 | 3 | 2 | 3 | 3 | 4 | 3 | 4 | 3 | 35 |
| 7 | P1: amorphisation–heating crossover model | 4 | 5 | 4 | 2 | 3 | 3 | 4 | 3 | 4 | 3 | 35 |
| 8 | P9: do uMLIP uncertainties flag PES softening | 4 | 4 | 3 | 2 | 3 | 3 | 3 | 4 | 4 | 4 | 34 |
| 9 | P12: early-warning signals in fairy-circle time series | 4 | 4 | 3 | 2 | 3 | 3 | 4 | 3 | 4 | 3 | 33 |
| 10 | P3: agent-independent EEG transition signature | 4 | 4 | 2 | 2 | 3 | 4 | 3 | 4 | 3 | 3 | 32 |
| 11 | X3: tight LR sextuples beyond 130 | 3 | 5 | 3 | 2 | 2 | 5 | 3 | 5 | 3 | 4 | 35 |
| 12 | P5: hidden-variable meta-analysis of Turkevich syntheses | 3 | 4 | 3 | 3 | 4 | 2 | 3 | 3 | 3 | 4 | 32 |
| 13 | P4: clock CpG causal status by Mendelian randomisation | 5 | 4 | 2 | 2 | 3 | 2 | 3 | 3 | 4 | 3 | 31 |
| 14 | P13: autonomic vs report placebo dissociation (re-analysis) | 4 | 3 | 2 | 2 | 3 | 2 | 3 | 3 | 3 | 3 | 28 |
| 15 | P6: lncRNA stress-induced stability (CLIP re-analysis) | 3 | 3 | 2 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 28 |
| 16 | P7: threshold between isodesmic and cooperative assembly | 3 | 3 | 2 | 1 | 2 | 2 | 3 | 4 | 3 | 3 | 26 |
| 17 | P11: transmission inference from long-term datasets | 4 | 4 | 3 | 1 | 2 | 2 | 3 | 3 | 3 | 3 | 28 |
| 18 | X7: tribology crossover re-analysis | 3 | 4 | 3 | 2 | 4 | 2 | 3 | 3 | 2 | 4 | 30 |
| 19 | P10: Erdős long-tail problems | 3 | 3 | 2 | 3 | 4 | 5 | 3 | 5 | 3 | 4 | 35 (crowded) |
| 20 | X8: OEIS LR sequences | 1 | 4 | 2 | 5 | 5 | 5 | 1 | 5 | 1 | 5 | 34 (low value) |

**Five most accessible:** 1, 2, 3, 5, 20. **Five high-impact:** 6, 7, 13, 1, 9. **Five overlooked:** 1 (below-ceiling regime),
2, 3, 11, 18. **Five emerging-science:** 1, 2, 5, 8, 19. **Three primary directions:** (A) rank 1; (B) rank 2 (folded into A as the
training-dynamics arm); (C) rank 3/4 (lonely-runner spectrum computation).

Note on honesty of ranking: the "T" and "R" columns dominate because they encode what *this* team can execute. A better-resourced
team should rank P2 and P1 higher.
