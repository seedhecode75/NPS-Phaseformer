"""Identical repaired objectives for every architecture; not claimed as novelty."""
import torch
from torch import nn
from torch.nn import functional as F

class RestorationLoss(nn.Module):
    def __init__(self, mode='charbonnier', weights=(.2741,.2222,.3357,.1680)):
        super().__init__();self.mode=mode
        self.register_buffer('weights',torch.tensor(weights))
        lap=torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])
        self.register_buffer('lap',lap[None,None].repeat(3,1,1,1))
        if mode=='paper_fixed':
            from pytorch_msssim import MS_SSIM
            from torchvision.models import vgg16,VGG16_Weights
            self.ms=MS_SSIM(data_range=1.,size_average=True,channel=3)
            self.vgg=vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval().requires_grad_(False)
            self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        elif mode!='charbonnier': raise ValueError(mode)

    def perceptual(self,p,t):
        p=F.interpolate((p-self.mean)/self.std,(224,224),mode='bilinear',align_corners=False)
        t=F.interpolate((t-self.mean)/self.std,(224,224),mode='bilinear',align_corners=False)
        loss=p.new_zeros(())
        for i,layer in enumerate(self.vgg):
            p=layer(p)
            with torch.no_grad(): t=layer(t)
            if i in (3,8,15,22):loss=loss+F.l1_loss(p,t)
        return loss

    def one(self,p,t):
        # Both tensors in physical [0,1] coordinates, prediction may overshoot.
        char=((p-t).square()+1e-6).sqrt().mean()
        if self.mode=='charbonnier':return char
        if min(p.shape[-2:])<=160:raise ValueError('MS-SSIM requires dimensions >160')
        grad=F.l1_loss(F.conv2d(p,self.lap,groups=3),F.conv2d(t,self.lap,groups=3))
        structural=1-self.ms(p.clamp(0,1),t)
        per=self.perceptual(p,t)
        return (torch.stack((char,grad,structural,per))*self.weights).sum()

    def forward(self, outputs, lo, hi):
        return .6*self.one((outputs[0]+1)/2,lo)+.4*self.one((outputs[1]+1)/2,hi)
