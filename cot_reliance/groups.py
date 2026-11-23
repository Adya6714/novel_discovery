"""Finite groups as multiplication tables. Element 0 is the identity."""
import itertools, numpy as np

def perm_group(perms):
    perms = [tuple(p) for p in perms]
    n = len(perms[0]); ident = tuple(range(n))
    perms.sort(key=lambda p: p != ident)  # identity first
    idx = {p: i for i, p in enumerate(perms)}
    G = len(perms); T = np.zeros((G, G), dtype=np.int64)
    for a, pa in enumerate(perms):
        for b, pb in enumerate(perms):
            T[a, b] = idx[tuple(pa[pb[i]] for i in range(n))]  # (a*b)(i)=a(b(i))
    return T

def parity(p):
    s = 0
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            s += p[i] > p[j]
    return s % 2

def cyclic(n):
    return np.add.outer(np.arange(n), np.arange(n)) % n

def get_group(name):
    if name.startswith('Z') and name[1:].isdigit():
        return cyclic(int(name[1:]))
    if name == 'A5':
        return perm_group([p for p in itertools.permutations(range(5)) if parity(p) == 0])
    if name == 'S3':
        return perm_group(list(itertools.permutations(range(3))))
    if name == 'S5':
        return perm_group(list(itertools.permutations(range(5))))
    if name == 'Z2xZ30':  # abelian non-cyclic, order 60
        els = [(a, b) for a in range(2) for b in range(30)]
        T = np.zeros((60, 60), dtype=np.int64)
        for i, (a, b) in enumerate(els):
            for j, (c, d) in enumerate(els):
                T[i, j] = els.index(((a+c) % 2, (b+d) % 30))
        return T
    raise ValueError(name)

if __name__ == '__main__':
    for g in ['Z60', 'A5', 'S3', 'Z2xZ30']:
        T = get_group(g); G = len(T)
        assert all(sorted(T[i]) == list(range(G)) for i in range(G))
        assert (T[0] == np.arange(G)).all()
        assoc = all(T[T[a, b], c] == T[a, T[b, c]] for a in range(G) for b in range(G) for c in range(0, G, 7))
        print(g, G, 'abelian' if (T == T.T).all() else 'nonabelian', 'assoc', assoc)
