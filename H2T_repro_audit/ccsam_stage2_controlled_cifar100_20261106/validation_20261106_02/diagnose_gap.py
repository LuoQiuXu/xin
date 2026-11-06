"""Three-path comparison on one identical frozen Clean feature map; no updates."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
import ccsam_common as c
from prepare_ccsam import load_official_train_pickle

def save(name, value):
    with (OUT/name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)

def comparison(value, reference):
    a, b = value.double(), reference.double()
    error = (a-b).abs()
    relative = error / b.abs().clamp_min(1e-12)
    return dict(max_abs=float(error.max()), max_relative=float(relative.max()),
        mean_abs=float(error.mean()), bitwise_equal=bool(torch.equal(value, reference)),
        allclose=bool(torch.allclose(value, reference, rtol=1e-5, atol=1e-6)),
        element_violations=int((error > (1e-6+1e-5*b.abs())).sum()),
        values_finite=bool(torch.isfinite(value).all()),
        reference_finite=bool(torch.isfinite(reference).all()),
        relative_denominator="max(abs(reference),1e-12)")

c.install_test_guard(c.DEFAULT_DATA_ROOT)
checkpoint = c.resolve_checkpoint()
c.seed_everything(42, deterministic=True)
device = torch.device("cuda:0")
backbone, classifier, metadata = c.build_models(checkpoint, device)
classifier.eval()
initial_backbone = c.state_hash(backbone)
initial_classifier = c.state_hash(classifier)
_, tx, _ = c.build_transforms()
raw, _, _, _ = load_official_train_pickle(c.DEFAULT_DATA_ROOT)
with np.load(c.STABILITY_RUN/"split_indices.npz", allow_pickle=False) as split:
    validation_ids = split["validation_raw_indices"].astype(np.int64)
batches, sample_rows = [], []
confirmed = False
with torch.no_grad():
    for start in range(0, len(validation_ids), 128):
        ids = validation_ids[start:start+128]
        images = torch.stack([tx(Image.fromarray(raw[int(i)])) for i in ids]).to(device)
        feature_map = backbone(images)
        assert feature_map.shape[1:] == (64,8,8) and feature_map.dtype == torch.float32
        a = classifier(feature_map)
        pooled_b = F.avg_pool2d(feature_map, feature_map.size(3))
        pooled_b = pooled_b.view(pooled_b.size(0), -1)
        pooled_c = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        b = classifier.module.fc(pooled_b)
        cc = classifier.module.fc(pooled_c)
        record = dict(validation_start=start, batch_size=len(ids), feature_shape=list(feature_map.shape),
            feature_dtype=str(feature_map.dtype), feature_finite=bool(torch.isfinite(feature_map).all()),
            feature_sha256=c.hash_tensor(feature_map), pool_C_vs_B=comparison(pooled_c, pooled_b),
            logits_B_vs_A=comparison(b,a), logits_C_vs_A=comparison(cc,a),
            input_sha256=c.hash_tensor(images))
        batches.append(record)
        for j in range(len(ids)):
            sample_rows.append(dict(validation_position=start+j,
                pool_max_abs=comparison(pooled_c[j],pooled_b[j])["max_abs"],
                pool_max_relative=comparison(pooled_c[j],pooled_b[j])["max_relative"],
                logits_B_vs_A_max_abs=comparison(b[j],a[j])["max_abs"],
                logits_C_vs_A_max_abs=comparison(cc[j],a[j])["max_abs"],
                logits_C_vs_A_max_relative=comparison(cc[j],a[j])["max_relative"],
                old_gate_allclose=bool(torch.allclose(cc[j],a[j],rtol=1e-5,atol=1e-6))))
        print(json.dumps(record), flush=True)
        if not record["logits_C_vs_A"]["allclose"]:
            snapshot = OUT/"tensor_snapshots"
            snapshot.mkdir(exist_ok=False)
            path = snapshot/"first_adaptive_failure.pt"
            with path.open("xb") as stream:
                torch.save(dict(feature_map=feature_map.cpu(), avg_gap=pooled_b.cpu(),
                    adaptive_gap=pooled_c.cpu(), logits_A=a.cpu(), logits_B=b.cpu(), logits_C=cc.cpu(),
                    classifier_state={k:v.detach().cpu() for k,v in classifier.state_dict().items()},
                    classifier_state_sha256=initial_classifier, validation_ids=torch.from_numpy(ids.copy()),
                    validation_start=start, checkpoint_sha256=c.EXPECTED_CLEAN_SHA256,
                    rng_state=torch.get_rng_state()), stream)
            confirmed = (record["feature_finite"] and record["logits_B_vs_A"]["bitwise_equal"]
                and record["pool_C_vs_B"]["values_finite"] and record["logits_C_vs_A"]["values_finite"]
                and record["pool_C_vs_B"]["max_abs"] > 0)
            break
assert c.state_hash(backbone)==initial_backbone and c.state_hash(classifier)==initial_classifier
assert c.sha256_file(checkpoint)==c.EXPECTED_CLEAN_SHA256
with (OUT/"GAP_BEFORE_PER_SAMPLE.csv").open("x",encoding="utf-8",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(sample_rows[0]))
    writer.writeheader();writer.writerows(sample_rows)
errors=np.array([r["logits_C_vs_A_max_abs"] for r in sample_rows])
result=dict(status="POOLING_NUMERIC_CAUSE_CONFIRMED" if confirmed else "CAUSE_UNDETERMINED",
    checkpoint_path=str(checkpoint), checkpoint_sha256=c.EXPECTED_CLEAN_SHA256,
    seed=metadata["seed"], epoch=metadata["epoch"], batches=batches,
    samples_checked=len(sample_rows), samples_failing_old_gate=sum(not r["old_gate_allclose"] for r in sample_rows),
    sample_logit_error_quantiles=dict(zip(["min","p50","p95","p99","max"],np.quantile(errors,[0,.5,.95,.99,1]).tolist())),
    backbone_and_bn_bitwise_unchanged=True,classifier_bitwise_unchanged=True,
    optimizer_updates=0, formal_training_started=False, test_guard=c.guard_pass(),
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32))
save("GAP_BEFORE.json",result)
print(json.dumps({k:v for k,v in result.items() if k!="batches"}),flush=True)
