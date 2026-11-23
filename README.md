# novel_discovery — an autonomous research programme (Nov 2026)

This repository holds the complete record of an AI-run research programme: a landscape audit of 13 research areas,
the selection of the directions that could actually be advanced with the available resources (4 CPU cores, no lab, no GPU),
and two original computational studies.

**Main result.** Transformers trained on *correct* chain-of-thought intermediates do not necessarily use them. On window-sum chains,
models ignore their written intermediates entirely when recomputing from the input is cheaper (K ≤ 2; 6/6 models), and read them when
reading is cheaper (K ≥ 4; 15/15). Which route is adopted is predicted by an independent cost measure: how fast each route is learned
when the other is masked out (6/6 comparisons). At near-parity the choice is seed-dependent (6/9 recompute). Self-correction of corrupted
chains is a by-product of the route choice. See `executive_summary.pdf` (5 pages) and `REPORT.md`. The principle (availability/simplicity bias)
is known; the demonstration for CoT intermediates is the contribution. Toy scale; AI-generated; not yet human-reviewed.

| Where | What |
|---|---|
| `REPORT.md` | Full research report (primary study) |
| `executive_summary.pdf` | 5-page executive summary (source: `summary/executive_summary.html`) |
| `docs/01_landscape_and_prioritisation.md` | Audit of the 13 programmes, expansion, ranked 20 opportunities |
| `docs/02_preregistration_cot_reliance.md` | Preregistration (frozen before any window-task model was trained; later design deviations D14–D16 are logged) |
| `docs/03_novelty_matrix.md` | Prior-art comparison |
| `docs/04_adversarial_review.md`, `docs/04_adversarial_review_raw.md` | Independent (AI) adversarial review, our responses and the outcomes of the experiments it prompted |
| `docs/05_decision_log.md` | Every decision, pivot, bug and deviation, in order |
| `docs/06_human_vs_ai_contribution.md` | Who did what |
| `docs/07_publication_readiness.md` | Self-assessment, failed hypotheses, future experiments |
| `docs/search_log.md` | Literature search log and limitations |
| `cot_reliance/` | Primary study: do chain-trained transformers read or recompute their written intermediates? |
| `lonely_runner/` | Secondary study: exhaustive Lonely Runner spectrum searches for n = 7, 8, 9 |
| `figures/` | All figures (regenerable) |

## Reproducing the primary study (`cot_reliance/`)
Requirements: Python ≥ 3.10, `torch` (CPU is enough), `numpy`, `matplotlib`. Total compute ≈ 6–8 CPU-core-hours.
```bash
cd cot_reliance
python3 test_targets.py                       # target alignment unit test
# train one model (K = window size; depth 2; seed 0)
python3 train.py --group iwin:11:4 --Lmin 12 --layers 2 --seed 0 --steps 3000 --out runs/sweep/K4_L2_s0.pt
./continue_until_gate.sh runs/sweep/K4_L2_s0.pt   # continues training until the accuracy gate passes (cap 18k steps)
# full sweep as run here
cat sweep_jobs_iL2.txt | xargs -P 4 -L 1 sh -c 'python3 train.py --group iwin:11:$0 --Lmin 12 --layers $2 --seed $1 --steps 3000 --out runs/sweep/K$0_L$2_s$1.pt'
P=2 ./watch_continue.sh sweep_jobs_iL2.txt
python3 analyse_sweep.py 'runs/sweep/*.pt' results/sweep_results.json --untrained
python3 summary_table.py                      # results/summary_table.md
# review-response experiments
./queue3.sh      # forced-route learnability (eval every 20 steps), position-offset control, extra K=3 seeds, dynamics
./queue4.sh      # spillover test (loss only on j > K)
./queue5.sh      # forced-route learnability with position shifts
python3 forced_analysis.py; python3 forced_analysis.py 'runs/forced_posoff/*.pt' results/forced_posoff_results.json
python3 census.py 'runs/sweep/K[234]_L2_s*.pt' 'runs/k3seeds/*.pt' 'runs/posoff/*.pt' 'runs/lossfrom/*.pt'
python3 dynamics.py
python3 test_oracle.py                        # metric validation on exact READ / RECOMPUTE oracles
python3 make_figures.py ../figures
```
`results/` contains the evaluated numbers for every model actually trained here, so the figures and tables can be checked without retraining.

## Reproducing the secondary study (`lonely_runner/`)
```bash
cd lonely_runner
gcc -O3 -march=native -o neartight neartight.c
python3 ml_exact.py                 # sanity checks against published values
./run_search.sh 6 110 4             # positive control: must give 348 (Cordella 2026)
./run_search.sh 7 100 4 && python3 summarise.py data/nt_n7_V100.txt
python3 crosscheck.py               # independent brute-force cross-check
```

## Status labels
Everything described as a result was executed in this repository. Nothing is a physical measurement. Proposed-but-unexecuted
work is labelled **[proposed]**.
