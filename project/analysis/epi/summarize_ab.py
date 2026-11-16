#!/usr/bin/env python3
"""Summarize interleaved A/B CSV lines from epi*.out."""
import math, sys, collections
from statistics import mean

def geo(xs):
  xs=[x for x in xs if x>0]
  return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float('nan')

path=sys.argv[1]
rows=[]
smokes=[]
for line in open(path, errors='replace'):
  line=line.strip()
  if line.startswith('SMOKE_'):
    smokes.append(line)
  if line.startswith('FAIL,'):
    rows.append(('FAIL', line))
    continue
  parts=line.split(',')
  if len(parts)>=9 and parts[0] in {'BK64','HYB2','HYB17','HYB22','HYB25','EPI0','EPI0b','EPI1','EPI2'} and parts[1].isdigit():
    tag,r,S,H,I,E,k,ms,err=parts[:9]
    rows.append((tag,int(r),int(S),int(H),int(I),int(E),int(k),float(ms),float(err)))

print('SMOKES:', *smokes, sep='\n  ')
by=collections.defaultdict(list)
err_by=collections.defaultdict(list)
for row in rows:
  if row[0]=='FAIL':
    print(row[1]); continue
  tag,r,S,H,I,E,k,ms,err=row
  by[(tag,S,H,I,E,k)].append(ms)
  err_by[(tag,S,H,I,E,k)].append(err)

tags=sorted({k[0] for k in by})
shapes=sorted({k[1:] for k in by})
print('tags', tags, 'shapes', len(shapes))
# per-shape means
means={k: mean(v) for k,v in by.items()}
base='HYB22' if any(t=='HYB22' for t in tags) else ('HYB17' if any(t=='HYB17' for t in tags) else 'HYB2')
cand=[t for t in tags if t.startswith('EPI')]
print(f'base={base} candidates={cand}')
for c in cand:
  ratios=[]; ratios2=[]; ratios64=[]; maxerr=0
  print(f'\n=== {c} vs {base} ===')
  for sh in shapes:
    if (c,*sh) not in means or (base,*sh) not in means: continue
    r=means[(c,*sh)]/means[(base,*sh)]
    ratios.append(r)
    e=max(err_by[(c,*sh)])
    maxerr=max(maxerr,e)
    print(f'  S={sh[0]} I={sh[2]} E={sh[3]} k={sh[4]}: {c}/{base}={r:.4f} ({(r-1)*100:+.2f}%) err={e:.6f}')
    if ('HYB2',*sh) in means:
      ratios2.append(means[(c,*sh)]/means[('HYB2',*sh)])
    if ('BK64',*sh) in means:
      ratios64.append(means[(c,*sh)]/means[('BK64',*sh)])
  if ratios:
    g=geo(ratios)
    print(f'GEO {c}/{base}={g:.4f} ({(g-1)*100:+.2f}%)  max_err={maxerr:.6f}  n={len(ratios)}')
  if ratios2:
    g=geo(ratios2); print(f'GEO {c}/HYB2={g:.4f} ({(g-1)*100:+.2f}%)')
  if ratios64:
    g=geo(ratios64); print(f'GEO {c}/BK64={g:.4f} ({(g-1)*100:+.2f}%)')
  keep = ratios and geo(ratios)<0.997 and maxerr < 0.2  # loose band; tune
  print('KEEP?' , keep)
