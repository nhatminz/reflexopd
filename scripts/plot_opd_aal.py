#!/usr/bin/env python3
"""Two curves using exact per-step counter ratios, never cumulative/moving AAL."""
import argparse
import csv
from pathlib import Path

def read(path):
    rows=list(csv.DictReader(Path(path).open()))
    x=[];y=[]
    for row in rows:
        rounds=float(row['step_verification_rounds']);accepted=float(row['step_accepted_tokens'])
        value=accepted/rounds if rounds else 0.
        if abs(value-float(row['step_aal']))>1e-9:raise ValueError('step_aal does not match exact step counters')
        x.append(int(row['step']));y.append(value)
    return x,y

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fastgrpo',required=True);p.add_argument('--opd-reflex',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('choose a new plot output')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label,path in (('FastGRPO',a.fastgrpo),('OPD Reflex',a.opd_reflex)):
        x,y=read(path);plt.plot(x,y,label=label)
    plt.xlabel('GRPO step');plt.ylabel('Step AAL (includes target bonus)');plt.legend();plt.grid(alpha=.2)
    output.parent.mkdir(parents=True,exist_ok=True);plt.savefig(output,bbox_inches='tight');plt.close()

if __name__=='__main__':main()
