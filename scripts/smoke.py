"""Single command, isolated temporary paths, no downloaded dataset required."""
import subprocess,sys,tempfile,json,shutil
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='phaseformer_smoke_') as tmp:
        t=Path(tmp);data=t/'data';manifest=t/'split.json';run=t/'run';ev=t/'eval'
        commands=[['scripts/make_smoke_data.py','--root',str(data)],
          ['scripts/prepare_data.py','--root',str(data),'--out',str(manifest),'--test-count','2','--val-count','2'],
          ['train.py','--manifest',str(manifest),'--variant','nps','--device','cpu','--size','32','--epochs','1','--max-steps','1','--output',str(run)],
          ['evaluate.py','--manifest',str(manifest),'--checkpoint',str(run/'best.pt'),'--size','32','--output',str(ev)]]
        for command in commands:subprocess.run([sys.executable,*command],cwd=root,check=True)
        result=json.loads((ev/'summary.json').read_text())
        assert result['count']==2 and result['checkpoint_status']=='smoke_only'
        print('END-TO-END SMOKE PASSED (synthetic data only)')
if __name__=='__main__':main()
