"""Independent cross-check of neartight.c: for every tuple at small radius, compute ML exactly with ml_exact.ML
(maximisation over all candidate times m/(v_i+v_j), a different method from the endpoint-intersection test in neartight.c)
and compare the set of primitive near-tight tuples (ML < 1/n) with the C kernel's output."""
import itertools, subprocess, sys
from fractions import Fraction
from math import gcd
from functools import reduce
from ml_exact import ML
for n, V in [(4, 40), (5, 22), (6, 16)]:
    ref = {tuple(v) for v in itertools.combinations(range(1, V + 1), n)
           if reduce(gcd, v) == 1 and ML(list(v))[0] < Fraction(1, n)}
    out = subprocess.run(['./neartight', str(n), str(V), '1', str(V)], capture_output=True, text=True).stdout
    got = {tuple(map(int, l.split())) for l in out.splitlines() if l.strip()}
    print(f'n={n} V={V}: brute-force {len(ref)} vs kernel {len(got)} -> {"IDENTICAL" if ref == got else "MISMATCH"}')
    assert ref == got
