#!/usr/bin/env python3
"""Share of 3- and 4-bit error patterns that a (72,64) Hsiao SEC-DED code miscorrects or misses.
Hsiao construction: 8 weight-1 check columns, all 56 weight-3 columns, 8 weight-5 columns chosen
to balance the row weights (seeded search). Usage: python3 secded_patterns.py"""
import itertools, random
r = 8
w1 = [1 << i for i in range(r)]
w3 = [sum(1 << i for i in c) for c in itertools.combinations(range(r), 3)]
w5all = [sum(1 << i for i in c) for c in itertools.combinations(range(r), 5)]
random.seed(1)
best = None
for _ in range(20000):
    w5 = random.sample(w5all, 8)
    rows = [sum((c >> i) & 1 for c in w3 + w5) for i in range(r)]
    if best is None or max(rows) - min(rows) < best[0]:
        best = (max(rows) - min(rows), w5)
cols = w1 + w3 + best[1]; n = len(cols); colset = set(cols)
tot3 = sum(1 for _ in itertools.combinations(range(n), 3))
mis3 = sum(1 for c in itertools.combinations(range(n), 3) if cols[c[0]] ^ cols[c[1]] ^ cols[c[2]] in colset)
tot4 = und4 = 0
for c in itertools.combinations(range(n), 4):
    tot4 += 1; und4 += (cols[c[0]] ^ cols[c[1]] ^ cols[c[2]] ^ cols[c[3]]) == 0
print(f"k=3 miscorrected (silent): {mis3/tot3:.3f}   k=4 undetected (q4): {und4/tot4:.4f}")
