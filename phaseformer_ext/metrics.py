import numpy as np
from skimage.metrics import structural_similarity
from .uiqm_compat import getUIQM

def quantize(x):return np.rint(np.clip(x,0,1)*255).astype(np.uint8)

def image_metrics(pred,target=None):
    p=pred.astype(np.float64)
    out={'uiqm_compat':float(getUIQM(p))}
    if target is not None:
        if pred.shape!=target.shape:raise ValueError('Shape mismatch: resizing during metric computation is forbidden')
        t=target.astype(np.float64)
        mse=float(np.mean((p-t)**2))
        # JSON-safe sentinel for a mathematically infinite PSNR.
        out['psnr_rgb']=float(10*np.log10(255**2/mse)) if mse>0 else None
        out['perfect_match']=mse==0
        out['ssim_rgb']=float(structural_similarity(p,t,data_range=255,channel_axis=2,
             gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
    return out
