"""Mechanistic evidence, not a restoration benchmark."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model_with_eca import inv_mag
from phaseformer_ext.modules import phase_descriptors
from phaseformer_ext.model import build_model,load_initial_weights
from phaseformer_ext.runtime import seed_all,dump_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/upstream_uieb_state.pt')
    p.add_argument('--out',default='results/mechanism_probe.json');a=p.parse_args();seed_all(123)
    x=torch.randn(2,8,32,32,dtype=torch.float64)+.2;phase=inv_mag(x)
    impulse=torch.zeros_like(x);impulse[...,0,0]=1
    dc_error=(phase.mean((-2,-1))-torch.sign(x.sum((-2,-1)))/1024).abs().max().item()
    baseline=build_model('baseline');proposed=build_model('nps')
    load_initial_weights(baseline,a.checkpoint);load_initial_weights(proposed,a.checkpoint)
    z=torch.rand(1,3,64,64)*2-1
    with torch.no_grad():diff=max((u-v).abs().max().item() for u,v in zip(baseline(z),proposed(z)))
    proposed(z)[0].square().mean().backward()
    result={'status':'synthetic_mechanism_and_checkpoint_compatibility_only','dc_identity_max_error':dc_error,
      'phase_statistic_random_mean':phase_descriptors(phase).mean((0,2)).tolist(),
      'phase_statistic_impulse_mean':phase_descriptors(impulse).mean((0,2)).tolist(),
      'pretrained_nps_zero_init_output_max_difference':diff,
      'correction_gradient_maxima':[proposed.attention[i].correction.weight.grad.abs().max().item() for i in range(3)]}
    dump_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
