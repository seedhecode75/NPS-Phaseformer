import os,random,json,platform,hashlib
from pathlib import Path
import numpy as np
import torch

def seed_all(seed,threads=1):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def environment():
    return {'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'threads':torch.get_num_threads()}

def dump_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False))

def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
