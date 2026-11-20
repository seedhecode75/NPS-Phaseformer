"""Paired image bootstrap. Seed-to-seed variation must be reported separately."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from phaseformer_ext.runtime import dump_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--proposed',required=True)
    p.add_argument('--out',default='results/comparison.json');p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args()
    bs=json.loads((Path(a.baseline)/'summary.json').read_text());ps=json.loads((Path(a.proposed)/'summary.json').read_text())
    for key in ('protocol','manifest_sha256','count'):
        if bs[key]!=ps[key]:raise ValueError(f'Unmatched evaluation protocol: {key}')
    if not bs['manifest_sha256']:raise ValueError('Paired comparisons require a manifest')
    def read(path):
        with open(Path(path)/'per_image.csv') as f:return {r['name']:r for r in csv.DictReader(f)}
    b,pred=read(a.baseline),read(a.proposed)
    if b.keys()!=pred.keys():raise ValueError('Different image sets')
    rng=np.random.default_rng(123);result={}
    for key in ('psnr_rgb','ssim_rgb','uiqm_compat'):
        bv=np.array([float(b[n][key]) for n in sorted(b)]);pv=np.array([float(pred[n][key]) for n in sorted(b)])
        delta=pv-bv
        boots=delta[rng.integers(0,len(delta),(a.bootstrap,len(delta)))].mean(1)
        result[key]={'baseline':float(bv.mean()),'proposed':float(pv.mean()),'paired_delta':float(delta.mean()),
                    'image_bootstrap_95ci':np.quantile(boots,[.025,.975]).tolist()}
    dump_json(a.out,{'metrics':result,'warning':'Image bootstrap is conditional on these trained models; it is not training-seed uncertainty.'})
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
