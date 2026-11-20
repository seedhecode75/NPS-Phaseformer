import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from phaseformer_ext.model import build_model,load_initial_weights,VARIANTS
from phaseformer_ext.data import PairedImages,verify_manifest
from phaseformer_ext.losses import RestorationLoss
from phaseformer_ext.runtime import seed_all,environment,dump_json,file_hash

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/controlled.json');p.add_argument('--manifest',required=True)
    p.add_argument('--variant',choices=VARIANTS,default='nps');p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=123)
    p.add_argument('--init');p.add_argument('--resume');p.add_argument('--epochs',type=int)
    p.add_argument('--size',type=int);p.add_argument('--max-steps',type=int)
    p.add_argument('--fixed-tau',action='store_true');p.add_argument('--scale',choices=['full','ac'],default='full')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if a.epochs is not None:cfg['epochs']=a.epochs
    if a.size is not None:cfg['size']=a.size
    if a.init and a.resume:raise ValueError('Use init OR resume')
    verify_manifest(a.manifest)
    seed_all(a.seed,cfg['threads']);out=Path(a.output)
    if (out/'last.pt').exists() and not a.resume:raise FileExistsError('Use a new output directory or --resume')
    out.mkdir(parents=True,exist_ok=True)
    model_args={'variant':a.variant,'tau':cfg['tau'],'learnable':not a.fixed_tau,'scale':a.scale}
    model=build_model(**model_args)
    init_info=load_initial_weights(model,a.init) if a.init else None
    model=model.to(a.device)
    lossfn=RestorationLoss(cfg['loss']).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
    train=PairedImages(a.manifest,'train',cfg['size'],True);val=PairedImages(a.manifest,'val',cfg['size'])
    # num_workers=0 deliberately makes RNG/resume auditable across platforms.
    loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=0)
    vloader=DataLoader(val,batch_size=1,shuffle=False,num_workers=0)
    metadata={'config':cfg,'model_args':model_args,'seed':a.seed,'manifest_sha256':file_hash(a.manifest),
      'environment':environment(),'initialization':init_info,'init_sha256':file_hash(a.init) if a.init else None,
      'status':'smoke_only' if a.max_steps else 'controlled_training_not_paper_reproduction'}
    # Reset data/augmentation RNG after model construction for paired-seed fairness.
    seed_all(a.seed,cfg['threads'])
    start,best,steps=0,float('-inf'),0
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=True)
        for key in ('config','model_args','seed','manifest_sha256'):
            if ck['metadata'][key]!=metadata[key]:raise ValueError(f'Resume mismatch: {key}')
        metadata=ck['metadata']
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start,best,steps=ck['epoch']+1,ck['best_psnr'],ck['steps']
        random.setstate(ck['rng']['python']);torch.set_rng_state(ck['rng']['torch'])
        if torch.cuda.is_available() and ck['rng']['cuda']:torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    dump_json(out/'config.json',metadata)
    with open(out/'train.jsonl','a' if a.resume else 'w') as log:
        for epoch in range(start,cfg['epochs']):
            model.train();total=0.;count=0
            for x,lo,hi,_ in loader:
                x,lo,hi=x.to(a.device),lo.to(a.device),hi.to(a.device)
                optimizer.zero_grad(set_to_none=True)
                loss=lossfn(model(x),lo,hi)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad'])
                optimizer.step();steps+=1;count+=1;total+=loss.item()
                if a.max_steps and steps>=a.max_steps:break
            model.eval();psnrs=[]
            with torch.inference_mode():
                for x,lo,_,_ in vloader:
                    pred=((model(x.to(a.device))[0]+1)/2).clamp(0,1)
                    mse=(pred-lo.to(a.device)).square().mean()
                    psnrs.append(float(-10*torch.log10(mse.clamp_min(1e-12))))
                    if a.max_steps:break
            score=float(np.mean(psnrs));improved=score>best;best=max(best,score);scheduler.step()
            ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'epoch':epoch,'best_psnr':best,'steps':steps,'metadata':metadata,
                'rng':{'python':random.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
            torch.save(ck,out/'last.pt')
            if improved:torch.save(ck,out/'best.pt')
            row={'epoch':epoch,'steps':steps,'train_loss':total/count,'val_psnr_float_rgb':score,'lr':scheduler.get_last_lr()[0]}
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if a.max_steps and steps>=a.max_steps:break
if __name__=='__main__':main()
