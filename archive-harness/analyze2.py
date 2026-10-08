import csv, sys, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
t = defaultdict(dict)
for r in rows:
    t[(r['shape'], int(r['round']))][r['arm']] = float(r['us'])
shapes = list(dict.fromkeys(r['shape'] for r in rows))
print(f"{'shape':28} {'base us':>9}  " + '  '.join(f"{p:>15}" for p in ['new/base','new/twin','newtwin/base','newtwin/twin']) + "   range")
for s in shapes:
    rounds = sorted(k[1] for k in t if k[0] == s)
    cells = []; meds = []
    for n_, b_ in [('new','base'),('new','twin'),('newtwin','base'),('newtwin','twin')]:
        ratios = [t[(s, r)][n_] / t[(s, r)][b_] for r in rounds]
        m = st.median(ratios); meds.append(m)
        cells.append(f"{m:.3f} ({sum(x<1 for x in ratios):2d}/{len(ratios)})")
    print(f"{s:28} {min(t[(s,r)]['base'] for r in rounds):9.1f}  " + '  '.join(f"{c:>15}" for c in cells) + f"   {min(meds):.3f}..{max(meds):.3f}")
