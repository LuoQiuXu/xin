"""Immutable-source adapters for the explicitly approved 19c8e357 weights."""
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path(r"D:\h2t_cross_domain_results\fish16_weight_cache_20261031_01")
ORIGINAL_WEIGHT = CACHE / "resnet50-19c8e357.pth"
ORIGINAL_SHA = "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"
SAFE_WEIGHT = CACHE / "extractor_tensor_state.pth"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resnet = load_source("_fish16_original_resnet", "H2T_official_clean/H2T-main/models/resnet.py")
methods = load_source("_fish16_original_methods", "H2T_official_clean/H2T-main/methods.py")
pif_source = load_source("_fish16_review_cifar", "PI_H2T_review/H2T-main/models/resnet_cifar.py")
review_methods = load_source("_fish16_review_methods", "PI_H2T_review/H2T-main/methods.py")
previous = sys.modules.get("methods")
sys.modules["methods"] = review_methods
try:
    pi_source = load_source("_fish16_review_pi", "PI_H2T_review/H2T-main/pi_h2t.py")
finally:
    if previous is None:
        del sys.modules["methods"]
    else:
        sys.modules["methods"] = previous


def source_hashes():
    names = ["H2T_official_clean/H2T-main/" + n for n in (
        "models/resnet.py", "methods.py", "train_stage1.py", "train_stage2.py")]
    names += ["PI_H2T_review/H2T-main/" + n for n in (
        "pi_h2t.py", "methods.py", "models/resnet_cifar.py", "PI_H2T_REVIEW_ZH.md")]
    return {n: sha(ROOT / n) for n in names}


def prepare_weights():
    """Drop only ImageNet FC; legacy BatchNorm counters use PyTorch's zero rule."""
    assert sha(ORIGINAL_WEIGHT) == ORIGINAL_SHA, "Wrong pretrained file; no fallback"
    assert not SAFE_WEIGHT.exists(), "Refuse to overwrite an existing conversion"
    original = torch.load(ORIGINAL_WEIGHT, map_location="cpu", weights_only=False)
    assert isinstance(original, OrderedDict)
    assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in original.items())
    native = resnet.resnet50_fe(pretrained=False)  # Allocate original architecture, no forward yet.
    target = native.state_dict()
    assert set(original) - set(target) == {"fc.weight", "fc.bias"}
    assert original["fc.weight"].shape == (1000, 2048) and original["fc.bias"].shape == (1000,)
    missing = sorted(set(target) - set(original))
    assert len(missing) == 53 and all(k.endswith(".num_batches_tracked") for k in missing)
    assert all(original[k].shape == target[k].shape for k in set(original) & set(target))
    assert all(torch.isfinite(v).all() for v in original.values())
    filtered = OrderedDict((k, v) for k, v in original.items() if k not in {"fc.weight", "fc.bias"})
    # Versionless legacy BN state gets num_batches_tracked=0 inside PyTorch's strict loader.
    result = native.load_state_dict(filtered, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    actual = native.state_dict()
    assert all(torch.equal(actual[k], original[k]) for k in set(original) & set(target))
    assert all(actual[k].item() == 0 and actual[k].dtype == torch.int64 for k in missing)
    with SAFE_WEIGHT.open("xb") as f:
        torch.save(actual, f)
    safe = torch.load(SAFE_WEIGHT, map_location="cpu", weights_only=True)
    assert all(torch.equal(actual[k], safe[k]) for k in actual)
    report = {"source_url": resnet.model_urls["resnet50"], "source_path": str(ORIGINAL_WEIGHT),
              "source_sha256": ORIGINAL_SHA, "source_keys": len(original),
              "extractor_keys": len(actual), "dropped_only": ["fc.weight", "fc.bias"],
              "legacy_zero_bn_counters": missing, "learned_or_floating_keys_filled": [],
              "shape_mismatches": [], "strict_load": True,
              "all_265_backbone_weight_and_float_buffer_tensors_exact": True,
              "safe_tensor_checkpoint": str(SAFE_WEIGHT), "safe_sha256": sha(SAFE_WEIGHT),
              "legacy_load_note": "weights_only=False only for exact official SHA; downstream uses weights_only=True"}
    with (CACHE / "ADAPTER_PROVENANCE.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


class Extractor(nn.Module):
    def __init__(self, use_pif):
        super().__init__()
        info = json.loads((CACHE / "ADAPTER_PROVENANCE.json").read_text(encoding="utf-8"))
        assert info["source_sha256"] == ORIGINAL_SHA and sha(SAFE_WEIGHT) == info["safe_sha256"]
        self.backbone = resnet.resnet50_fe(pretrained=False)
        self.backbone.load_state_dict(torch.load(SAFE_WEIGHT, map_location="cpu", weights_only=True), strict=True)
        self.pif = pif_source.PermutationInvariantFusion() if use_pif else nn.Identity()

    def forward(self, x):
        return self.pif(self.backbone(x))


class FishModel(nn.Module):
    def __init__(self, use_pif=False, seed=42):
        super().__init__()
        # Isolate extra PIF parameter initialization so common backbone/head are identical.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.extractor = Extractor(False)
            self.classifier = resnet.Classifier(2048, 16).fc
            if use_pif:
                self.extractor.pif = pif_source.PermutationInvariantFusion()
        self.use_pif = use_pif

    def forward(self, x):
        return self.classifier(self.extractor(x).mean((2, 3)))

    def freeze(self):
        self.eval()
        self.extractor.requires_grad_(False)
        self.extractor.eval()
        self.classifier.requires_grad_(True)
        self.classifier.train()


def h2t_with_indices(recipient, donor):
    rng = torch.get_rng_state()
    selected = torch.randperm(recipient.shape[1])[:int(.25 * recipient.shape[1])]
    torch.set_rng_state(rng)
    return methods.H2T(recipient.clone(), donor, rho=.25), selected


def stage2_features(recipient, donor, labels, classifier, mode):
    if mode == "crt":
        return recipient.mean((2, 3)), None
    if mode == "h2t":
        fused, indices = h2t_with_indices(recipient, donor)
        return fused.mean((2, 3)), indices
    if mode == "pi":
        return pi_source.h2tf(recipient, donor, labels, classifier.weight,
                             mode="paper_theoretical_bounds", ratio_detach=True)
    raise ValueError(mode)


def state_sha(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    print(json.dumps(prepare_weights(), indent=2))
