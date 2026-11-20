import copy,math
import numpy as np
import pytest
import torch
from model_with_eca import inv_mag, ECA
from phaseformer_ext.modules import ReliabilityPhase,phase_descriptors,PhaseStatisticSkip
from phaseformer_ext.model import build_model,VARIANTS
from phaseformer_ext.metrics import image_metrics

torch.set_num_threads(1)
@pytest.mark.parametrize('shape',[(2,3,16,24),(1,2,15,17)])
def test_dc_collapse(shape):
    x=torch.randn(shape,dtype=torch.float64)+.3
    expected=torch.sign(x.sum((-2,-1)))/(shape[-1]*shape[-2])
    torch.testing.assert_close(inv_mag(x).mean((-2,-1)),expected,rtol=1e-10,atol=1e-12)

def test_equal_magnitude_distinct_phase():
    torch.manual_seed(4);x=torch.randn(1,2,16,16,dtype=torch.float64)
    p=inv_mag(x);impulse=torch.zeros_like(p);impulse[...,0,0]=1
    torch.testing.assert_close(torch.fft.fft2(p).abs(),torch.fft.fft2(impulse).abs())
    assert (phase_descriptors(p)-phase_descriptors(impulse)).abs().max()>.2
    torch.testing.assert_close(p.square().sum((-2,-1)),impulse.square().sum((-2,-1)))

def test_descriptor_cyclic_shift_invariance():
    x=torch.randn(1,3,16,16)
    torch.testing.assert_close(phase_descriptors(x),phase_descriptors(torch.roll(x,(3,7),(-2,-1))))

@pytest.mark.parametrize('kind',['zero','constant','tiny','random'])
def test_reliability_finite_gradients(kind):
    x={'zero':torch.zeros(1,3,16,24),'constant':torch.ones(1,3,16,24),
       'tiny':1e-12*torch.randn(1,3,16,24),'random':torch.randn(1,3,16,24)}[kind].requires_grad_()
    m=ReliabilityPhase();y=m(x);y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert torch.isfinite(m.raw_tau.grad).all()

def test_ra_symmetry_scale_limit():
    torch.manual_seed(3);x=torch.randn(1,2,15,17,dtype=torch.float64)
    m=ReliabilityPhase(tau=1e-5).double();z=m.spectrum(x)
    assert torch.fft.ifft2(z).imag.abs().max()<1e-12
    torch.testing.assert_close(m(x),inv_mag(x),atol=1e-7,rtol=1e-5)
    m=ReliabilityPhase().double()
    torch.testing.assert_close(m(x),m(x*7),atol=1e-12,rtol=1e-10)

def test_nps_exact_initialization_and_learning():
    torch.manual_seed(2);old=ECA(16);new=PhaseStatisticSkip(copy.deepcopy(old))
    x=torch.randn(2,16,16,16,requires_grad=True)
    torch.testing.assert_close(old(x),new(x),atol=0,rtol=0)
    new(x).square().mean().backward()
    assert new.correction.weight.grad.abs().max()>0

@pytest.mark.parametrize('variant',VARIANTS)
def test_full_model_backward(variant):
    torch.manual_seed(1);m=build_model(variant);x=torch.randn(1,3,16,24,requires_grad=True)
    y=m(x);assert y[0].shape==(1,3,16,24) and y[1].shape==(1,3,32,48)
    sum(t.square().mean() for t in y).backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

def test_baseline_and_nps_same_initialization():
    torch.manual_seed(42);a=build_model('baseline')
    torch.manual_seed(42);b=build_model('nps');x=torch.randn(1,3,16,16)
    with torch.no_grad():
        for ya,yb in zip(a(x),b(x)):torch.testing.assert_close(ya,yb,atol=0,rtol=0)

def test_cpu_bfloat16_module():
    x=torch.randn(1,3,16,16).bfloat16().requires_grad_();m=ReliabilityPhase()
    with torch.autocast('cpu',dtype=torch.bfloat16):y=m(x)
    y.float().square().mean().backward();assert torch.isfinite(x.grad).all()

def test_metrics_known_offset_and_constant():
    a=np.full((32,32,3),100,np.uint8);b=a+10;r=image_metrics(a,b)
    assert abs(r['psnr_rgb']-20*math.log10(255/10))<1e-10
    assert np.isfinite(r['uiqm_compat'])
    perfect=image_metrics(a,a);assert perfect['ssim_rgb']==1 and perfect['perfect_match']


def test_algebra_only_control():
    from phaseformer_ext.modules import UnitPhase
    for x in (torch.zeros(1,2,16,16),torch.ones(1,2,16,16),torch.randn(1,2,16,16)):
        torch.testing.assert_close(UnitPhase()(x),inv_mag(x),atol=1e-7,rtol=1e-5)
