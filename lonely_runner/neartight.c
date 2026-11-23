/* Exhaustive search for near-tight n-tuples: distinct positive integers v_1<...<v_n<=V with
   ML(v) < 1/n, where ML(v) = max_t min_i ||t v_i||.
   Exactness: G_i = {t : ||t v_i|| >= 1/n} is a finite union of closed intervals [(a+1/n)/v_i,(a+1-1/n)/v_i].
   The intersection G = cap_i G_i is closed and periodic; if nonempty, the left endpoint of any of its
   components is a left endpoint (a+1/n)/v_i of some G_i. By symmetry t -> 1-t we may restrict to points in (0,1/2]
   (a component with left endpoint > 1/2 is the mirror image of one with right endpoint < 1/2, whose left endpoint is < 1/2).
   Point t = (n a + 1)/(n v_i) lies in G_j iff ((n a+1) v_j mod n v_i) in [v_i, (n-1) v_i].
   Hence ML(v) < 1/n  iff  no candidate point lies in all G_j.
   Usage: neartight n V v1lo v1hi   (enumerates tuples with v1 in [v1lo,v1hi]); prints primitive near-tight tuples. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define MAXN 10
#define MAXP 200000
typedef struct { int64_t num, den, vi; } pt;
static int n, V; static int v[MAXN];
static pt pool[MAXN+1][MAXP]; static int cnt[MAXN+1];
static long long leaves = 0, found = 0;
static inline int in_good(const pt *p, int64_t vj) {
    int64_t r = (p->num * vj) % p->den;
    return r >= p->vi && r <= (n - 1) * p->vi;
}
static int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }
static void dfs(int k) {  /* k elements placed: v[0..k-1]; pool[k] = surviving candidate points */
    if (k == n) {
        leaves++;
        if (cnt[k] == 0) {
            int g = 0; for (int i = 0; i < n; i++) g = gcd(g, v[i]);
            if (g == 1) { found++; for (int i = 0; i < n; i++) printf("%d%c", v[i], i == n - 1 ? '\n' : ' '); }
        }
        return;
    }
    int lo = (k == 0) ? 1 : v[k - 1] + 1;
    for (int x = lo; x <= V - (n - 1 - k); x++) {
        v[k] = x; int c = 0;
        if (k == n - 1) {               /* leaf: early exit as soon as one point survives */
            int surv = 0;
            for (int s = 0; s < cnt[k] && !surv; s++) if (in_good(&pool[k][s], x)) surv = 1;
            if (surv) { leaves++; continue; }
            for (int64_t a = 0; 2 * (n * a + 1) <= n * (int64_t)x && !surv; a++) {
                pt q = { n * a + 1, (int64_t)n * x, x }; int ok = 1;
                for (int j = 0; j < k && ok; j++) ok = in_good(&q, v[j]);
                if (ok) surv = 1;
            }
            if (surv) { leaves++; continue; }
            cnt[k + 1] = 0; dfs(k + 1); continue;
        }
        for (int s = 0; s < cnt[k]; s++) if (in_good(&pool[k][s], x)) pool[k + 1][c++] = pool[k][s];
        for (int64_t a = 0; 2 * (n * a + 1) <= n * (int64_t)x; a++) {
            pt q = { n * a + 1, (int64_t)n * x, x }; int ok = 1;
            for (int j = 0; j < k && ok; j++) ok = in_good(&q, v[j]);
            if (ok) { if (c >= MAXP) { fprintf(stderr, "overflow\n"); exit(1); } pool[k + 1][c++] = q; }
        }
        cnt[k + 1] = c; dfs(k + 1);
    }
}
int main(int argc, char **argv) {
    n = atoi(argv[1]); V = atoi(argv[2]); int lo = atoi(argv[3]), hi = atoi(argv[4]);
    cnt[0] = 0;
    for (int x = lo; x <= hi && x <= V - n + 1; x++) {
        v[0] = x; int c = 0;
        for (int64_t a = 0; 2 * (n * a + 1) <= n * (int64_t)x; a++) { pt q = { n * a + 1, (int64_t)n * x, x }; pool[1][c++] = q; }
        cnt[1] = c; dfs(1);
    }
    fprintf(stderr, "n=%d V=%d v1=[%d,%d] leaves=%lld primitive_near_tight=%lld\n", n, V, lo, hi, leaves, found);
}
