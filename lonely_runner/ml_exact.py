"""Exact maximum loneliness ML(v) = max_t min_i ||t v_i|| with rational arithmetic.
The function t -> min_i ||t v_i|| is piecewise linear; its local maxima occur where an increasing piece
t v_i - a meets a decreasing piece b - t v_j, i.e. at t = m/(v_i+v_j) (i = j allowed, giving tent peaks).
So ML = max over pairs (i<=j) and m in [0, v_i+v_j] of min_l ||m v_l/(v_i+v_j)||."""
import sys
from fractions import Fraction
from math import gcd

def ML(v):
    best = Fraction(0); arg = None
    dens = sorted({a + b for i, a in enumerate(v) for b in v[i:]})
    for D in dens:
        for m in range(1, D // 2 + 1):   # symmetry t -> 1 - t
            val = min(min(m * x % D, D - m * x % D) for x in v)
            if val * best.denominator > best.numerator * D:
                best = Fraction(val, D); arg = Fraction(m, D)
    return best, arg

def analyse(v):
    n = len(v); ml, t = ML(v); p, q = ml.numerator, ml.denominator
    return dict(v=v, ML=f'{p}/{q}', p=p, q=q, k=q - n * p, t=str(t), maxv_over_q=max(v) / q)

if __name__ == '__main__':
    # sanity: known values
    assert ML([1, 2, 3, 4, 5])[0] == Fraction(1, 6)
    assert ML([1, 3, 4, 5, 9])[0] == Fraction(1, 6)          # tight quintuple (BHK)
    assert ML([8, 4*0+3, 4*0+11, 4*0+19])[0] == Fraction(7, 30)  # Fan-Sun family r=0: (2r+7)/(4(2r+7)+2)
    assert ML([8, 4*5+3, 4*5+11, 4*5+19])[0] == Fraction(17, 70)
    assert ML([5, 6, 11, 17, 23, 28])[0] == Fraction(8, 51)  # Fan-Sun n=6, k=3
    assert ML([1, 3, 4, 5, 18, 46])[0] == Fraction(4, 25)    # Cordella Table 2
    assert ML([1, 2, 3, 4, 5, 7, 18])[0] == Fraction(3, 23)  # Cordella sec 6.3, k=2 for n=7
    print('sanity OK', file=sys.stderr)
