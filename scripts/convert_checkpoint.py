"""Convert the audited upstream full-module pickle via a restricted allowlist."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import model_with_eca as original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args()
    classes = [set, original.Restormer, original.MDTA, original.ECA, original.TransformerBlock,
        original.GDFN, original.DownSample, original.UpSample, original.UpSample1,
        nn.ModuleList, nn.Sequential, nn.Conv1d, nn.Conv2d, nn.LayerNorm,
        nn.PixelShuffle, nn.PixelUnshuffle, nn.AdaptiveAvgPool2d, nn.Sigmoid]
    with torch.serialization.safe_globals(classes):
        model = torch.load(a.input, map_location='cpu', weights_only=True)
    if not isinstance(model, original.Restormer):
        raise TypeError('Expected upstream Restormer')
    # Reject a pickle whose topology differs from the checked source constructor.
    baseline = original.Restormer()
    baseline.load_state_dict(model.state_dict(), strict=True)
    dst = Path(a.output); dst.parent.mkdir(parents=True, exist_ok=True)
    meta = {'source_sha256': hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
            'source_commit': '6e759dc782e514c985672c545205dea86a3d21b5',
            'variant': 'baseline', 'status': 'upstream_pretrained_not_retrained'}
    torch.save({'model': model.state_dict(), 'metadata': meta}, dst)
    dst.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))

if __name__ == '__main__': main()
