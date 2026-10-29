"""Minimal executable smoke test; no real checkpoint or data needed."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
def main():
    root=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);C=100
        counts=np.array([500]*35+[50]*35+[10]*30)
        coarse=np.arange(C)%20
        # Artificial mapping: one supported, one unsupported Few class.
        coarse[70]=coarse[0]
        coarse[71]=99
        ids=np.arange(6,dtype=np.int64);labels=np.array([70,70,70,71,71,71],dtype=np.int64)
        logits=np.zeros((6,C),dtype=np.float32)
        for i,y in enumerate(labels):logits[i,y]=1
        alt=logits.copy();alt[0,70]=0;alt[0,1]=2
        np.savez(d/"base.npz",ids=ids,labels=labels,logits=logits)
        np.savez(d/"h2t.npz",ids=ids,labels=labels,logits=alt)
        (d/"m.json").write_text(json.dumps({"class_counts":counts.tolist(),"coarse_labels":coarse.tolist()}))
        cmd=[sys.executable,str(root/"evaluate.py"),"--no-fusion",str(d/"base.npz"),
             "--h2t",str(d/"h2t.npz"),"--metadata",str(d/"m.json"),"--output",str(d/"out")]
        r=subprocess.run(cmd,capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        result=json.loads((d/"out"/"summary.json").read_text())
        assert result["supported"]["observed_classes"]==1
        assert result["unsupported"]["observed_classes"]==1
        assert result["supported"]["mean_class_gain_pp"]<0
        fail=subprocess.run(cmd,capture_output=True,text=True)
        assert fail.returncode!=0,"must refuse overwrite"
        print("PASS positive path and output protection")
if __name__=="__main__":main()
