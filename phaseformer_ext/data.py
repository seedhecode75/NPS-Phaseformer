import hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def image_map(root):
    paths = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in EXTENSIONS)
    result = {}
    for p in paths:
        if p.stem in result: raise ValueError(f'Duplicate image stem: {p.stem}')
        result[p.stem] = p
    if not result: raise ValueError(f'No images in {root}')
    return result


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pil_tensor(im):
    return torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1).copy()) / 255

class PairedImages(Dataset):
    def __init__(self, manifest, split, size=256, augment=False):
        self.document = json.loads(Path(manifest).read_text())
        self.rows = self.document['splits'][split]
        self.root = Path(self.document['root'])
        self.size, self.augment = size, augment
        if not self.rows: raise ValueError(f'Empty {split} split')

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x = Image.open(self.root / row['input']).convert('RGB')
        y = Image.open(self.root / row['target']).convert('RGB')
        if self.augment:
            if random.random() < 0.5:
                x,y=x.transpose(Image.Transpose.FLIP_LEFT_RIGHT),y.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                x,y=x.transpose(Image.Transpose.FLIP_TOP_BOTTOM),y.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        resample=Image.Resampling.BICUBIC
        x=pil_tensor(x.resize((self.size,self.size),resample))*2-1
        lo=pil_tensor(y.resize((self.size,self.size),resample))
        hi=pil_tensor(y.resize((2*self.size,2*self.size),resample))
        return x,lo,hi,row['name']


def verify_manifest(manifest):
    doc=json.loads(Path(manifest).read_text()); root=Path(doc['root'])
    names=set();digests={}
    for split,rows in doc['splits'].items():
        for row in rows:
            if row['name'] in names:raise ValueError('Repeated image across manifest')
            names.add(row['name'])
            for field in ('input','target'):
                actual=sha256(root/row[field])
                if actual!=row[field+'_sha256']:raise ValueError(f"Changed dataset file: {row[field]}")
                key=(field,actual)
                if key in digests and digests[key]!=split:raise ValueError('Cross-split duplicate content')
                digests[key]=split
    return {k:len(v) for k,v in doc['splits'].items()}
