"""Paired evaluation or unpaired inference; lossless RGB PNG metrics."""
import argparse,json,time,csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from phaseformer_ext.model import build_model,load_initial_weights,VARIANTS
from phaseformer_ext.data import PairedImages,image_map,pil_tensor,verify_manifest
from phaseformer_ext.metrics import image_metrics,quantize
from phaseformer_ext.runtime import seed_all,environment,dump_json,file_hash

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--variant',choices=VARIANTS)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--manifest');group.add_argument('--inputs')
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--size',type=int,default=256)
    p.add_argument('--head',choices=['low','high'],default='low');p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();seed_all(a.seed,a.threads)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);meta=ck.get('metadata',{})
    model_args=meta.get('model_args',{'variant':meta.get('variant','baseline')}).copy()
    if a.variant:
        if 'model_args' in meta and a.variant!=model_args['variant']:raise ValueError('Variant differs from trained checkpoint')
        model_args['variant']=a.variant
    model=build_model(**model_args);load_info=load_initial_weights(model,a.checkpoint)
    model=model.to(a.device).eval();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.size%8 or a.noise<0:raise ValueError('size must be divisible by 8; noise nonnegative')
    # Same corruptions despite variant-specific parameter initialization draws.
    seed_all(a.seed,a.threads)
    if a.manifest:
        verify_manifest(a.manifest)
        ds=PairedImages(a.manifest,a.split,a.size)
        iterator=(ds[i] for i in range(len(ds)))
    else:
        pairs=image_map(a.inputs)
        iterator=((pil_tensor(Image.open(f).convert('RGB').resize((a.size,a.size),Image.Resampling.BICUBIC))*2-1,None,None,n) for n,f in pairs.items())
    rows=[];head=0 if a.head=='low' else 1
    with torch.inference_mode():
        for x,lo,hi,name in iterator:
            if a.noise:
                # Noise is specified in [0,1] intensity units, input stored [-1,1].
                x=(x+2*a.noise*torch.randn_like(x)).clamp(-1,1)
            x=x.unsqueeze(0).to(a.device)
            pred=((model(x)[head][0]+1)/2).cpu().permute(1,2,0).numpy()
            q=quantize(pred);Image.fromarray(q).save(out/(name+'.png'))
            target=lo if head==0 else hi
            target=None if target is None else quantize(target.permute(1,2,0).numpy())
            row={'name':name,**image_metrics(q,target)};rows.append(row)
    with open(out/'per_image.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics={k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in ('psnr_rgb','ssim_rgb','uiqm_compat') if any(r.get(k) is not None for r in rows)}
    summary={'count':len(rows),'metrics':metrics,'protocol':{'size':a.size,'head':a.head,'noise':a.noise,
       'pixels':'RGB uint8 rounded, clipped; bicubic resize before inference; no crop',
       'ssim':'skimage Gaussian sigma=1.5 population covariance', 'uiqm':'FUSION compatibility variant; not certified equal to paper scorer'},
       'checkpoint_sha256':file_hash(a.checkpoint),'manifest_sha256':file_hash(a.manifest) if a.manifest else None,
       'model_args':model_args,'load':load_info,'environment':environment(),'checkpoint_status':meta.get('status'),
       'warning':'Architecture changed without training' if load_info['missing_new_parameters'] else None}
    dump_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
