"""Generate synthetic pairs for software tests ONLY, never benchmark claims."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='tmp/smoke_data');a=p.parse_args()
    root=Path(a.root)
    for s in ('raw-890','reference-890'):(root/s).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    for i in range(8):
        clean=rng.integers(0,256,(48,64,3),dtype=np.uint8)
        degraded=(clean.astype(float)*np.array([.5,.8,.9])+np.array([10,20,30])).clip(0,255).astype('uint8')
        Image.fromarray(clean).save(root/'reference-890'/f'{i}.png')
        Image.fromarray(degraded).save(root/'raw-890'/f'{i}.png')
if __name__=='__main__':main()
