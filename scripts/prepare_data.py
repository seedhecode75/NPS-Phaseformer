"""Create disjoint immutable paired-image manifests; no silent filename drops."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phaseformer_ext.data import image_map, sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True); p.add_argument('--input',default='raw-890')
    p.add_argument('--target',default='reference-890'); p.add_argument('--out',default='splits/uieb.json')
    p.add_argument('--test-count',type=int,default=90); p.add_argument('--val-count',type=int,default=80)
    p.add_argument('--test-list'); p.add_argument('--seed',type=int,default=2025)
    a=p.parse_args(); root=Path(a.root).resolve()
    x,y=image_map(root/a.input),image_map(root/a.target)
    if x.keys()!=y.keys(): raise ValueError(f'Unmatched stems: {set(x)^set(y)}')
    names=sorted(x); rng=random.Random(a.seed)
    if a.test_list:
        test=[Path(s.strip()).stem for s in Path(a.test_list).read_text().splitlines() if s.strip()]
        if len(test)!=len(set(test)) or not set(test)<=set(names): raise ValueError('Invalid test list')
        remaining=sorted(set(names)-set(test));rng.shuffle(remaining)
    else:
        rng.shuffle(names);test=names[:a.test_count];remaining=names[a.test_count:]
    if not 0<a.val_count<len(remaining): raise ValueError('Need nonempty train and validation splits')
    groups={'test':test,'val':remaining[:a.val_count],'train':remaining[a.val_count:]}
    # Identical files in different splits can leak even under different names.
    seen={}; splits={}
    for split,items in groups.items():
        rows=[]
        for n in sorted(items):
            hx,hy=sha256(x[n]),sha256(y[n])
            for kind,digest in [('input',hx),('target',hy)]:
                key=(kind,digest)
                if key in seen and seen[key]!=split: raise ValueError(f'Content leakage: {n}')
                seen[key]=split
            rows.append({'name':n,'input':str(x[n].relative_to(root)), 'target':str(y[n].relative_to(root)),
                         'input_sha256':hx,'target_sha256':hy})
        splits[split]=rows
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite split: {out}')
    out.write_text(json.dumps({'root':str(root),'seed':a.seed,'protocol':'new_controlled_split_not_paper_split','splits':splits},indent=2))
    print(json.dumps({k:len(v) for k,v in splits.items()}))
if __name__=='__main__':main()
