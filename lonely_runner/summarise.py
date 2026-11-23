import sys, json, collections
from fractions import Fraction
from ml_exact import analyse
rows = [analyse(list(map(int, l.split()))) for l in open(sys.argv[1]) if l.strip()]
n = len(rows[0]['v']) if rows else 0
ks = collections.Counter(r['k'] for r in rows)
mx = max(rows, key=lambda r: r['maxv_over_q'])
out = sys.argv[2] if len(sys.argv) > 2 else None
if out: json.dump(rows, open(out, 'w'))
print(f"{sys.argv[1]}: n={n} count={len(rows)} k-dist={dict(sorted(ks.items()))} max(maxv/q)={Fraction(max(mx['v']), mx['q'])}={mx['maxv_over_q']:.4f} at {mx['v']} ML={mx['ML']}")
viol = [r for r in rows if not (1 <= r['k'] <= n)]
print('amended Fan-Sun violations (k outside [1,n]):', viol[:10])
