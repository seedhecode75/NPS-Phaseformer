"""Launch a paired-seed study. No test-set selection is performed."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--config',default='configs/controlled.json');p.add_argument('--output',default='runs')
    p.add_argument('--device',default='cuda');p.add_argument('--seeds',type=int,nargs='+',default=[123,456,789])
    p.add_argument('--variants',nargs='+',default=['baseline','nps','ra_qk','ra_nps','spatial_control'])
    a=p.parse_args()
    for seed in a.seeds:
        for v in a.variants:
            out=Path(a.output)/f'{v}_seed{seed}'
            subprocess.run([sys.executable,'train.py','--config',a.config,'--manifest',a.manifest,
               '--variant',v,'--seed',str(seed),'--output',str(out),'--device',a.device],check=True)
            subprocess.run([sys.executable,'evaluate.py','--manifest',a.manifest,'--checkpoint',str(out/'best.pt'),
               '--output',str(out/'test'),'--device',a.device],check=True)
if __name__=='__main__':main()
