"""Keep a dataset's provided test split intact; reserve validation from train."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phaseformer_ext.data import image_map,sha256,verify_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--train-input',default='train/input');p.add_argument('--train-target',default='train/target')
    p.add_argument('--test-input',default='test/input');p.add_argument('--test-target',default='test/target')
    p.add_argument('--val-count',type=int,default=150);p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--out',default='splits/ufo120.json');a=p.parse_args();root=Path(a.root).resolve()
    def pairs(xdir,ydir,prefix):
        x,y=image_map(root/xdir),image_map(root/ydir)
        if x.keys()!=y.keys():raise ValueError('Unmatched stems')
        return [{'name':prefix+n,'input':str(x[n].relative_to(root)),'target':str(y[n].relative_to(root)),
                 'input_sha256':sha256(x[n]),'target_sha256':sha256(y[n])} for n in sorted(x)]
    train=pairs(a.train_input,a.train_target,'train_');test=pairs(a.test_input,a.test_target,'test_')
    if not 0<a.val_count<len(train):raise ValueError('Need nonempty train/validation')
    random.Random(a.seed).shuffle(train)
    doc={'root':str(root),'seed':a.seed,'protocol':'provided_test_split_validation_from_train',
         'splits':{'train':train[a.val_count:],'val':train[:a.val_count],'test':test}}
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,indent=2))
    try:print(verify_manifest(out))
    except Exception:
        out.unlink();raise
if __name__=='__main__':main()
