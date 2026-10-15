"""Independently load Stage II best/final states in the original evaluator."""
import argparse
import contextlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch

from formal_stage1 import dump, sha

BASE = Path(__file__).resolve().parent
RUNTIME = BASE/"runtime"
sys.path.insert(0, str(RUNTIME))


class Metrics(logging.Handler):
    def __init__(self):
        super().__init__()
        self.value = None
    def emit(self, record):
        m = re.search(r"\* Acc@1 ([\d.]+)% Acc@5 ([\d.]+)% HAcc ([\d.]+)% MAcc ([\d.]+)% TAcc ([\d.]+)%", record.getMessage())
        if m:
            self.value = dict(zip(["overall", "top5", "head", "medium", "tail"], map(float,m.groups())))


def evaluate(folder, arm, kind):
    evidence = folder/arm
    status = json.loads((evidence/"status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed" and status["completed_epochs"] == 10
    name = "model_best_stage2.pth.tar" if kind == "best" else "model_final.pth.tar"
    checkpoint_path = Path(status["model_dir"])/name
    assert sha(checkpoint_path) == status["checkpoints_sha256"][name]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == (status["stage2_best_epoch"] if kind == "best" else 10)
    assert all(torch.isfinite(v).all() for key in ["state_dict_model", "state_dict_classifier"] for v in checkpoint[key].values())
    spec = importlib.util.spec_from_file_location("original_stage2_eval", RUNTIME/"eval-modifed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_load, real_logger, real_validate = torch.load, module.create_logger, module.validate
    captures = {"class_counts": np.zeros(100,dtype=np.int64), "correct_counts": np.zeros(100,dtype=np.int64), "loads": []}
    metrics = Metrics()

    def load(path, *args, **kwargs):
        assert Path(path).resolve() == checkpoint_path.resolve()
        captures["loads"].append(str(path))
        return real_load(path, *args, **kwargs)

    def logger(cfg, name):
        result = real_logger(cfg,name)
        result[0].addHandler(metrics)
        captures["official_log"] = str(Path(result[2]).resolve().parent/"logs"/(cfg.name+".txt"))
        return result

    def validate(loader, model, classifier, lws_model, criterion, cfg, log, block=None):
        before = {k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        class Loader:
            def __len__(self):
                return len(loader)
            def __iter__(self):
                for images,targets in loader:
                    captures["targets"] = targets.numpy()
                    yield images,targets
        def hook(_module,_inputs,outputs):
            target = captures["targets"]
            prediction = outputs.argmax(1).detach().cpu().numpy()
            captures["class_counts"] += np.bincount(target,minlength=100)
            captures["correct_counts"] += np.bincount(target[target==prediction],minlength=100)
        handle = classifier.register_forward_hook(hook)
        try:
            result = real_validate(Loader(),model,classifier,lws_model,criterion,cfg,log,block)
        finally:
            handle.remove()
        after = {k:v.detach().cpu() for k,v in model.named_buffers()}
        assert all(torch.equal(v,after[k]) for k,v in before.items())
        captures["official_acc1"] = float(result[0])
        captures["ece"] = float(result[1])
        return result

    os.chdir(RUNTIME)
    sys.argv = ["eval-modifed.py", "--cfg", "config/cifar100/cifar100_imb001_stage2_ce_H2T.yaml",
        "data_path", str(BASE.parent/"H2T/data/cifar100"), "resume", str(checkpoint_path),
        "H2T", str(status["H2T"]), "name", status["run_name"]+"_"+kind+"_eval"]
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.load, module.create_logger, module.validate = load,logger,validate
    with (evidence/(kind+"_eval.stdout_stderr.log")).open("w",encoding="utf-8",buffering=1) as stream:
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            try:
                module.main()
            finally:
                torch.load = real_load
    assert len(captures["loads"]) == 1 and np.all(captures["class_counts"]==100)
    correct,counts = captures["correct_counts"],captures["class_counts"]
    per_class = correct/counts*100
    values = {"overall": float(correct.sum()/10000*100), "head": float(per_class[:36].mean()),
              "medium": float(per_class[36:71].mean()), "tail": float(per_class[71:].mean())}
    epoch = int(checkpoint["epoch"])
    original = status["epochs"][epoch-1]["eval_metrics_percent_rounded_3dp"]
    assert all(abs(values[k]-metrics.value[k])<.0006 and abs(values[k]-original[k])<.0006 for k in values)
    dump(evidence/(kind+"_evaluation.json"), {"arm": arm,"kind": kind,"epoch": epoch,
        "checkpoint": {"path": str(checkpoint_path),"sha256":sha(checkpoint_path)},
        "metrics_percent": values,"ece_percent": captures["ece"],
        "official_metrics_percent_rounded_3dp":metrics.value,"official_log":captures["official_log"],
        "class_counts":counts.tolist(),"correct_counts":correct.tolist(),"per_class_accuracy_percent":per_class.tolist(),
        "bn_buffers_unchanged_during_eval":True,"matches_training_epoch_evaluation":True,"test_samples":10000})
    print(f"{arm.upper()} {kind.upper()} EVAL PASS: epoch {epoch}; {values}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-id",required=True)
    parser.add_argument("--arm",choices=["dr","h2t"],required=True)
    parser.add_argument("--kind",choices=["best","final"],required=True)
    args = parser.parse_args()
    evaluate(BASE/"stage2_runs"/args.suite_id,args.arm,args.kind)
