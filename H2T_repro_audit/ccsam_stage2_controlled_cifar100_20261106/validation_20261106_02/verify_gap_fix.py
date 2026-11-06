"""Check the fixed helper on the saved identical failing feature map, without updates."""
import json
import sys
from pathlib import Path
import torch

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
import ccsam_common as c
c.install_test_guard(c.DEFAULT_DATA_ROOT)
path=OUT/"tensor_snapshots"/"first_adaptive_failure.pt"
snapshot=torch.load(path,map_location="cpu",weights_only=True)
checkpoint=c.resolve_checkpoint()
c.seed_everything(42,deterministic=True)
backbone,classifier,_=c.build_models(checkpoint,torch.device("cuda:0"))
classifier.eval()
backbone_before=c.state_hash(backbone)
classifier_before=c.state_hash(classifier)
assert classifier_before==snapshot["classifier_state_sha256"]
x=snapshot["feature_map"].cuda()
with torch.no_grad():
    a=classifier(x)
    gap=c.clean_compatible_gap(x)
    b=classifier.module.fc(gap)
assert gap.shape==(len(x),64) and gap.dtype==x.dtype==torch.float32
assert torch.equal(gap.cpu(),snapshot["avg_gap"])
assert torch.equal(a.cpu(),snapshot["logits_A"])
assert torch.equal(b,a)
assert torch.allclose(b,a,rtol=1e-5,atol=1e-6)
bad_shapes=[(2,64,8,7),(2,63,8,8),(2,64)]
for shape in bad_shapes:
    try:
        c.clean_compatible_gap(torch.empty(shape))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Malformed feature shape was accepted: "+str(shape))
assert c.state_hash(backbone)==backbone_before and c.state_hash(classifier)==classifier_before
result=dict(status="PASS",feature_shape=list(x.shape),feature_dtype=str(x.dtype),
    gap_shape=list(gap.shape),gap_dtype=str(gap.dtype),
    original_avg_vs_fixed_gap_max_abs=float((gap.cpu()-snapshot["avg_gap"]).abs().max()),
    original_A_vs_fixed_B_logits_max_abs=float((b-a).abs().max()),
    original_A_vs_fixed_B_logits_max_relative=float(((b.double()-a.double()).abs()/a.double().abs().clamp_min(1e-12)).max()),
    original_A_vs_fixed_B_bitwise_equal=True,allclose=True,rtol=1e-5,atol=1e-6,
    finite=bool(torch.isfinite(gap).all() and torch.isfinite(a).all() and torch.isfinite(b).all()),
    input_and_original_logits_match_saved_pre_fix=True,
    invalid_shapes_rejected=[list(s) for s in bad_shapes],
    checkpoint_sha256=c.sha256_file(checkpoint),checkpoint_expected=c.EXPECTED_CLEAN_SHA256,
    feature_snapshot_sha256=c.sha256_file(path),
    backbone_and_bn_bitwise_unchanged=True,classifier_bitwise_unchanged=True,
    optimizer_updates=0,formal_training_started=False,test_guard=c.guard_pass())
assert result["checkpoint_sha256"]==c.EXPECTED_CLEAN_SHA256
with (OUT/"GAP_AFTER.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
