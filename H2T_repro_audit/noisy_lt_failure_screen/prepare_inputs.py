"""Official CIFAR-N provenance + pure-integer adapter; no training or TEST read."""
import hashlib
import json
from pathlib import Path
import pickle
import urllib.request

import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
OUT = TASK / "results/cifar100n_lt_seed42_20261029_01"
CACHE = OUT / "primary_cache"
CLEAN = ROOT / "H2T_repro_audit/clean_vs_runs/clean_s1_seed42_vs_seed123_20261013"
OFFICIAL_URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-100_human.pt"
GIT_BLOB = "5dac9db8af8f7224f982fbac9794c67f60f8bd45"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


def main():
    source = CACHE / "downloads/CIFAR-100_human.pt"
    assert not (OUT / "INPUT_ATTESTATION.json").exists()
    if not source.exists():
        with urllib.request.urlopen(OFFICIAL_URL, timeout=45) as response:
            payload = response.read(2_000_000)
            final_url = response.url
        assert len(payload) == 1000959
        assert hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() == GIT_BLOB
        with source.open("xb") as f:
            f.write(payload)
    else:
        payload = source.read_bytes()
        final_url = OFFICIAL_URL
        assert hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() == GIT_BLOB
    import torch
    numeric = [np.ndarray, np.dtype, np._core.multiarray._reconstruct, np._core.multiarray.scalar,
               (np._core.multiarray._reconstruct,"numpy.core.multiarray._reconstruct"),
               (np._core.multiarray.scalar,"numpy.core.multiarray.scalar")]
    numeric += [type(np.dtype(t)) for t in [np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,np.float32,np.float64,np.bool_]]
    with torch.serialization.safe_globals(numeric):
        human = torch.load(source,map_location="cpu",weights_only=True)
    assert "clean_label" in human and "noisy_label" in human
    hc, hn = np.asarray(human["clean_label"]), np.asarray(human["noisy_label"])
    assert hc.shape == hn.shape == (50000,) and hc.dtype.kind in "iu" and hn.dtype.kind in "iu"
    assert ((hn>=0)&(hn<100)).all()
    train_path = ROOT / "H2T/data/cifar100/cifar-100-python/train"
    assert sha(train_path) == "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"
    with train_path.open("rb") as f:
        raw = pickle.load(f,encoding="latin1")
    clean = np.asarray(raw["fine_labels"],dtype=np.int64)
    assert np.array_equal(clean,hc), "STOP: official human clean labels do not match all50000 Python TRAIN labels"
    with np.load(CLEAN / "split_indices.npz",allow_pickle=False) as z:
        tr, va = z["train_raw_indices"].copy(), z["validation_raw_indices"].copy()
    assert len(tr) == 10847 and len(va) == 1980 and np.intersect1d(tr,va).size == 0
    expected_counts = np.array([int(500*.01**(c/99)) for c in range(100)])
    rng = np.random.RandomState(0)
    rebuilt=[]
    for c,n in enumerate(expected_counts):
        ids=np.flatnonzero(clean==c);rng.shuffle(ids);rebuilt.extend(ids[:n])
    assert np.array_equal(rebuilt,tr)
    assert np.array_equal(np.bincount(clean[va],minlength=100),[0]+[20]*99)
    pixels = raw["data"].reshape(-1,3,32,32).transpose(0,2,3,1)
    assert np.array_equal(np.load(CLEAN / "train_images.npy",allow_pickle=False,mmap_mode="r"),pixels[tr])
    assert np.array_equal(np.load(CLEAN / "validation_images.npy",allow_pickle=False,mmap_mode="r"),pixels[va])
    train_hashes = {hashlib.sha256(im.tobytes()).digest() for im in pixels[tr]}
    assert all(hashlib.sha256(im.tobytes()).digest() not in train_hashes for im in pixels[va])
    inputs = CACHE / "inputs"
    inputs.mkdir(exist_ok=False)
    np.save(inputs / "raw_clean_labels.npy",clean,allow_pickle=False)
    with (inputs / "human_labels.npz").open("xb") as f:
        np.savez_compressed(f,clean_label=hc.astype(np.int64),noisy_label=hn.astype(np.int64))
    with (inputs / "split.npz").open("xb") as f:
        np.savez_compressed(f,train_ids=tr,val_ids=va)
    np.save(inputs / "lt_images.npy",pixels[tr],allow_pickle=False)
    np.save(inputs / "validation_images.npy",pixels[va],allow_pickle=False)
    write_json(OUT / "INPUT_ATTESTATION.json",dict(
        official_repository="https://github.com/UCSC-REAL/cifar-10-100n",download_url=OFFICIAL_URL,final_download_url=final_url,
        official_git_blob_sha1=GIT_BLOB,human_file_bytes=source.stat().st_size,human_file_sha256=sha(source),
        official_human_keys=list(human),weights_only=True,compatibility="Closed primitive NumPy safe globals including old numpy.core aliases; no weights_only=False",
        human_clean_matches_Python_TRAIN_all50000=True,order="Python/PyTorch (not TFDS ordered file)",
        train_file_sha256=sha(train_path),historical_split_sha256=sha(CLEAN / "split_indices.npz"),
        train_ids_exact_seed0=True,train_val_id_overlap=0,train_val_exact_pixel_overlap=0,
        raw_train_pixel_cache_matches=True,validation_pixel_cache_matches=True,
        input_sha256={p.name:sha(p) for p in inputs.iterdir()},
        full_noise_fraction=float((hc!=hn).mean()),LT_noise_fraction=float((hc[tr]!=hn[tr]).mean()),
        no_official_TEST_open=True,trained=False))
    print(f"INPUT PASS: 50000/50000 clean labels align; full noise={(hc!=hn).mean():.6%}, LT noise={(hc[tr]!=hn[tr]).mean():.6%}",flush=True)


if __name__ == "__main__":
    main()
