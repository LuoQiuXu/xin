# PI-H2T failure scenario screen
Scope: one falsifiable question; test distribution sensitivity of **existing** checkpoints, without changing the fusion method or training.

- `screen.py` is runnable and checks paired IDs/labels, finite logits, shapes, and evaluation groups.
- Codex must implement minimal checkpoint inference using the existing repository model and fixed TRAIN-only holdout; **do not fabricate checkpoint availability**.
- Create `manifest.json` only once all referenced files exist. Example:
```json
{"class_counts":[500,482,464],"reference":"no_fusion","candidate":"pi_local",
 "conditions":["clean","noise_003","noise_008","blur_05","blur_10"],
 "methods":{"no_fusion":{"clean":"pred/no_clean.npz","noise_003":"pred/no_n003.npz","noise_008":"pred/no_n008.npz","blur_05":"pred/no_b05.npz","blur_10":"pred/no_b10.npz"},
 "pi_local":{"clean":"pred/pi_clean.npz","noise_003":"pred/pi_n003.npz","noise_008":"pred/pi_n008.npz","blur_05":"pred/pi_b05.npz","blur_10":"pred/pi_b10.npz"}}}
```
The three-class example is **schema illustration only**, not CIFAR-100 data. Use a length-100 class_counts vector in the real run. Paths are relative to manifest location. NPZ keys: ids, labels, logits; same IDs and labels for all arms. No pickle/object IDs. No clean/shift retraining or tuning. Comparison requires matched Stage-I and evaluation protocols.

Execution instruction: `CODEX_PI_FAILURE_SCENARIO_SCREEN.txt`.
