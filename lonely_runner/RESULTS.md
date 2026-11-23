# Lonely Runner spectrum: extended exhaustive searches (secondary direction)

## Question
For distinct positive integers v_1..v_n, ML(v) = max_t min_i ||t v_i||. A tuple is *near-tight* if ML < 1/n. Writing ML = p/q in
lowest terms and k = q − np:
* Kravitz's Loneliness Spectrum Conjecture: k = 1 for every near-tight tuple (false for n = 4, Fan & Sun).
* Fan & Sun's amended conjecture: 1 ≤ k ≤ n for every near-tight tuple.
* Cordella (arXiv 2609.03444) searched n=4 ≤ 200, n=5 ≤ 130, n=6 ≤ 110, n=7 ≤ 50, n=8 ≤ 40.
* Yi, Gill & Bu (arXiv 2610.40694, 28 Oct 2026 — found by our adversarial reviewer, missed by our own search) settle n = 4 completely
  and prove, for every n, an effective bound beyond which near-tight tuples lie on 2-D subtori of loneliness 1/n (conditional statements as
  stated there). This substantially reduces the informativeness of bounded searches such as ours.

## Method
`neartight.c`: exact integer test. Let G_i = {t : ||t v_i|| ≥ 1/n}, a union of closed intervals with left endpoints
(a + 1/n)/v_i. ML < 1/n iff ∩G_i = ∅; a nonempty closed periodic intersection contains the left endpoint of one of its components,
which is a left endpoint of some G_i, so it suffices to test the points t = (na+1)/(n v_i) ∈ (0, 1/2]:
t ∈ G_j ⇔ ((na+1) v_j mod n v_i) ∈ [v_i, (n−1) v_i]. A depth-first enumeration keeps only the candidate points that survive the
speeds placed so far. `ml_exact.py`: exact ML by maximising over t = m/(v_i+v_j) (local maxima of a min of tent functions occur where an
increasing piece meets a decreasing piece), with `fractions.Fraction`.
Only primitive tuples (gcd 1) are reported. Every run's leaf count is checked to equal C(V, n) (complete enumeration). `crosscheck.py` compares the kernel with an independent
brute-force exact-ML computation over *all* tuples for (n,V) = (4,40), (5,22), (6,16): identical sets (174, 18, 12). The adversarial
reviewer independently re-derived the exactness argument and reproduced (4,40) and (5,26).

## Validation (positive controls) — all reproduce Cordella exactly
| n | radius | Cordella | ours |
|---|---|---|---|
| 4 | 200 | 4125 primitive near-tight; 3836 k=1, 289 k=2 | 4125; 3836 / 289 |
| 5 | 130 | 64, all k=1 | 64, all k=1 |
| 6 | 110 | 348; 315 k=1, 33 k=3; max v_max/q = 105/43 | 348; 315 / 33; 105/43 at (6,31,37,68,99,105) |
| 7 | 50 | 29; 26 k=1, 3 k=2 | 29; 26 / 3 |
| 8 | 40 | 29, all k=1 | 29, all k=1 |
Known values checked in `ml_exact.py`: ML(1..5)=1/6, ML(1,3,4,5,9)=1/6, Fan–Sun family (8,4r+3,4r+11,4r+19) for r=0,5, ML(5,6,11,17,23,28)=8/51,
ML(1,3,4,5,18,46)=4/25, ML(1,2,3,4,5,7,18)=3/23.

## New results (beyond published radii)
| n | radius | tuples enumerated | primitive near-tight | k distribution | distinct ML values |
|---|---|---|---|---|---|
| 7 | 100 | 16,007,560,800 | 40 | k=1: 36, k=2: 4 | 1/8, 3/23*, 2/15, 5/37*, 3/22, 4/29, 9/65*, 5/36, …, 14/99 (*k=2) |
| 8 | 70 | 9,440,350,920 | 40 | k=1: 40 | s/(8s+1), s = 1..9 |
| 9 | 55 | 6,358,402,050 | 13 | k=1: 13 | s/(9s+1), s = 1..6, 8 |

Data: `data/nt_n{7,8,9}_V*.txt` (tuples) and `.json` (ML, k, optimal time).

**Findings.**
1. No counterexample to the amended Fan–Sun conjecture (1 ≤ k ≤ n) for n = 7 up to speed 100, n = 8 up to 70, n = 9 up to 55.
2. For n = 8 (≤ 70) and n = 9 (≤ 55) every near-tight tuple satisfies Kravitz's *original* conjecture (k = 1).
3. A new k = 2 septuple beyond Cordella's radius: (2, 7, 9, 11, 13, 20, 54), ML = 9/65 (optimal time 11/65). The four k = 2 values for n = 7
   are s/(7s+2) with s ∈ {3, 5, 9} (s = 3 attained twice); s = 7 (7/51) does not occur up to 100.
4. Tight septuples (ML = 1/8) up to 100: (1..7), (1,2,3,4,5,7,12), (1,4,5,6,7,11,13) only — these are *known* tight instances (Goddyn–Wong
   type; tight-instance literature), so this is a completeness statement within our radius, not a discovery; tight octuples up to 70:
   (1..8) only; tight 9-tuples up to 55: (1..9) only. Most n = 8, 9 hits are the trivial families (1,…,n−1, n·s) and their analogues over
   tight (n−1)-tuples.
5. Largest max v_i / q: n=7: 13/5 at (1,4,5,6,7,11,39); n=8: 21/17; n=9: 54/55.

**Failed hypothesis (recorded honestly).** After n = 4..8 we noticed that k > 1 occurred for n ∈ {4, 6, 7} with k = 2, 3, 2, which fits
"k ∈ {1, 2} when n ≡ 1, k ∈ {1, 3} when n ≡ 0 (mod 3), k = 1 when n ≡ 2". This already conflicts with n = 3 (Kravitz: k = 1 only) and the
n = 9 search (≡ 0 mod 3) found no k = 3 tuple up to 55, although for n = 6 the smallest k = 3 tuple has all speeds ≤ 17. We regard the
pattern as refuted in its naive form.

## Assessment
This is a verified data contribution (independent implementation, complete enumerations, exact arithmetic), not a theorem.
After Yi–Gill–Bu (2610.40694) and given that Cordella's public code could be scaled to the same radii, its significance is low: It extends the
evidence for the amended conjecture and for k = 1 at n = 8, 9, and supplies a new sporadic k = 2 septuple. it is suitable as
supplementary data for the structural programme of Giri–Kravitz–Jain–Cordella–Yi–Gill–Bu, not as a standalone paper.
