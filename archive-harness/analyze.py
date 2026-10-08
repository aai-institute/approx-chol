import csv, sys, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
t = defaultdict(dict)
for r in rows:
    t[(r['shape'], int(r['round']))][r['arm']] = float(r['us'])
shapes = list(dict.fromkeys(r['shape'] for r in rows))
arms = ['base_ctl', 'twin', 'new', 'newtwin']
print(f"{'shape':28} {'base us':>10} " + ' '.join(f"{a:>16}" for a in arms))
for s in shapes:
    rounds = sorted(k[1] for k in t if k[0] == s)
    base = [t[(s, r)]['base'] for r in rounds]
    cells = []
    for a in arms:
        ratios = [t[(s, r)][a] / t[(s, r)]['base'] for r in rounds]
        wins = sum(x < 1 for x in ratios)
        cells.append(f"{st.median(ratios):.3f} ({wins:2d}/{len(ratios)})")
    print(f"{s:28} {min(base):10.1f} " + ' '.join(f"{c:>16}" for c in cells))
