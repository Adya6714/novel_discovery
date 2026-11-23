# Research decision log (chronological)

| # | Decision | Reason / evidence |
|---|---|---|
| D1 | Audit resources before choosing problems | No lab, no GPU, HF blocked; only computational research is executable |
| D2 | Requested literature hosts (arXiv, OpenAlex, Crossref, S2, EuropePMC, OEIS) via a new personal environment | arXiv DNS-blocked; a prior-art audit without full texts would not be credible |
| D3 | Rejected Erdős long-tail problems as primary | Being mined at scale by AI labs in 2025–26 (≈100 resolved); high collision risk |
| D4 | Rejected "prove LRC for n+1 runners" | Verified up to 15 runners by Sep 2026 by specialised groups |
| D5 | Selected LR *spectrum* computations (n = 6,7,8) as secondary | Cordella (Sep 2026) lists open questions with explicit computational radii (n=7 ≤ 50, n=8 ≤ 40) that our kernel can extend massively |
| D6 | Selected read-vs-recompute of CoT intermediates as primary | Gap between "Ceiling of a Task" (arXiv 2609.33134; necessity above ceiling) and noise-driven unfaithfulness (arXiv 2602.01017): below-ceiling, correct-chain strategy selection is untested |
| D7 | Lonely-runner kernel validated against 3 published counts (4125, 64, 348) and k-splits (3836/289; 315/33), max ratio 105/43 | Positive controls before any new claim |
| D8 | Abandoned group word problem (Z60 vs A5) as main CoT task | Pilot chain models still at loss 2.7–3.3 after 3000 steps (60×60 table memorisation too slow for budget); direct models at chance (ln 60). Replaced by window-sum task with a continuous cost dial K |
| D10 | **Bug found and fixed:** training targets were shifted by one (position trained to predict the chain value two steps ahead). Caught by the preregistered clean-accuracy gate (K=2 model: train loss 0.0016 but next-step accuracy 0.09 = chance). Fixed in `window_task.py` and `task.py`; added `test_targets.py`. All models trained before the fix (group pilot, `wpilot_*`) are invalid and excluded; the group-pilot abandonment (D8) stands on its slow-learning grounds only as a qualitative observation | Integrity |
| D11 | Deviation from preregistration: added K = 5 to the K grid; steps fixed at 4000 for depth-2 models (corrected pilot: loss 1e-4, clean acc 1.00 at 4000 for K = 2 and 12) | Resolution near the expected crossover; logged before sweep results were seen |
| D12 | Lonely runner n=7 radius 100 completed (40 primitive near-tight, k ∈ {1,2}); n=8 radius 70 launched | Extends Cordella's 50 / 40 |
| D9 | Preregistered window-task hypotheses before training any window model | `docs/02_preregistration_cot_reliance.md` |
