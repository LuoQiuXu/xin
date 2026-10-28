"""Small synthetic contract check, no CIFAR or GPU required."""
import json
import tempfile
from pathlib import Path
import subprocess
import sys
import numpy as np

def main():
    root=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp); y=np.array([0,0,1,1,2,2],dtype=np.int64)
        ids=np.arange(6,dtype=np.int64)
        base=np.zeros((6,3));base[np.arange(6),y]=2
        alternate=base.copy();alternate[0]=[0,3,0]
        specs={}
        for arm, prediction in (("no_fusion",base),("pi_local",alternate)):
            specs[arm]={}
            for cond in ("clean","noise_003"):
                filename=f"{arm}_{cond}.npz"
                np.savez_compressed(d/filename,ids=ids,labels=y,logits=prediction)
                specs[arm][cond]=filename
        manifest={"class_counts":[500,40,5],"methods":specs,"reference":"no_fusion",
                  "candidate":"pi_local","conditions":["clean","noise_003"]}
        (d/"manifest.json").write_text(json.dumps(manifest),encoding="utf8")
        p=subprocess.run([sys.executable,str(root/"screen.py"),"--manifest",str(d/"manifest.json"),
                          "--output",str(d/"results")],capture_output=True,text=True)
        assert p.returncode==0,p.stderr
        assert (d/"results"/"paired_contrasts.csv").exists()
        assert (d/"results"/"summary.json").exists()
        np.savez_compressed(d/"pi_local_noise_003.npz",ids=ids[::-1],labels=y,logits=alternate)
        failed=subprocess.run([sys.executable,str(root/"screen.py"),"--manifest",str(d/"manifest.json"),
                               "--output",str(d/"bad")],capture_output=True,text=True)
        assert failed.returncode!=0,"unpaired IDs must fail"
    print("PASS: positive analysis and invalid pairing rejection")
if __name__=="__main__":main()
