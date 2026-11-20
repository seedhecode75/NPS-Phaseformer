"""Measured latency and parameters; explicitly partial FLOP accounting."""
import argparse,time,statistics,json,math
from pathlib import Path
import torch
from torch import nn
from model_with_eca import MDTA,ECA
from phaseformer_ext.modules import ReliableMDTA,ReliabilitySkip,PhaseStatisticSkip
from phaseformer_ext.model import build_model,VARIANTS
from phaseformer_ext.runtime import seed_all,environment,dump_json

def measure(variant,size,device,threads,warmup,runs):
    seed_all(123,threads);m=build_model(variant).to(device).eval();x=torch.rand(1,3,size,size,device=device)*2-1
    counts={'conv_matmul_flops':0,'fft_pair_estimate':0};handles=[]
    def conv(mod,args,out):
        kernel=math.prod(mod.kernel_size); per=mod.in_channels//mod.groups*kernel
        counts['conv_matmul_flops']+=2*out.numel()*per
    def attn(mod,args,out):
        b,c,h,w=args[0].shape
        counts['conv_matmul_flops']+=4*b*mod.num_heads*(c//mod.num_heads)**2*h*w
        counts['fft_pair_estimate']+=2*10*b*c*h*w*math.log2(h*w)
    def skip(mod,args,out):
        b,c,h,w=args[0].shape
        counts['fft_pair_estimate']+=10*b*c*h*w*math.log2(h*w)
    for mod in m.modules():
        if isinstance(mod,(nn.Conv1d,nn.Conv2d)):handles.append(mod.register_forward_hook(conv))
        if isinstance(mod,(MDTA,ReliableMDTA)):handles.append(mod.register_forward_hook(attn))
        if isinstance(mod,(ECA,ReliabilitySkip,PhaseStatisticSkip)):handles.append(mod.register_forward_hook(skip))
    with torch.inference_mode():m(x)
    for h in handles:h.remove()
    active=set();hooks=[]
    def used(mod,args,out):active.update(id(p) for p in mod.parameters(recurse=False))
    for mod in m.modules():hooks.append(mod.register_forward_hook(used))
    with torch.inference_mode():m(x)
    for h in hooks:h.remove()
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):m(x)
        sync()
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        timings=[]
        for _ in range(runs):
            sync();t=time.perf_counter();m(x);sync();timings.append((time.perf_counter()-t)*1000)
    return {'variant':variant,'parameters_registered':sum(p.numel() for p in m.parameters()),
      'parameters_active':sum(p.numel() for p in m.parameters() if id(p) in active),
      'conv_matmul_gflops':counts['conv_matmul_flops']/1e9,'fft_gflops_estimate':counts['fft_pair_estimate']/1e9,
      'latency_ms_median':statistics.median(timings),'latency_ms_min':min(timings),'latency_ms_samples':timings,
      'images_per_second':1000/statistics.median(timings),
      'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['baseline','unit_qk','nps','ra_qk','ra_nps'])
    p.add_argument('--size',type=int,default=256);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--runs',type=int,default=20);p.add_argument('--out',default='results/benchmark.json')
    a=p.parse_args()
    if a.size%8 or a.runs<1:raise ValueError('size multiple of 8; runs > 0')
    rows=[measure(v,a.size,a.device,a.threads,a.warmup,a.runs) for v in a.variants]
    result={'status':'measured_random_weight_execution_not_accuracy','environment':environment(),'size':a.size,'batch':1,'precision':'float32','outputs':'both heads computed','warmup':a.warmup,'runs':a.runs,
      'flop_convention':'2 FLOPs per multiply-accumulate; conv and attention matmul only. FFT separately estimated at 10*C*N*log2(N) per full FFT/IFFT pair. Elementwise, pooling, nonlinearities and normalization excluded. Not paper-comparable total GFLOPs.', 'rows':rows}
    dump_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
