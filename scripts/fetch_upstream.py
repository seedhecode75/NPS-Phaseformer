"""Fetch the pinned original Phaseformer UIEB checkpoint and convert it locally."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/Mdraqibkhan/Phaseformer/6e759dc782e514c985672c545205dea86a3d21b5/checkpoints/UIEB/best.pth'
EXPECTED = 'fbb2092e299e774e9fc35a4ddb6a24a0a4ac159e9c5b754b3f1d6dff18f9dfc8'
if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'best.pth'
        urlretrieve(URL, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError('Upstream checkpoint checksum mismatch')
        subprocess.run([sys.executable, str(ROOT/'scripts/convert_checkpoint.py'),
                        '--input', str(path), '--output', str(ROOT/'checkpoints/upstream_uieb_state.pt')], check=True)
