# Cross-domain H2T / PI-H2T failure-boundary study: data gate

## Research question

We are not trying to prove that H2T or PI-H2T is universally weak. We want to identify the conditions under which their reported head-to-tail transfer benefit does not generalize across visual domains, then explain the condition responsible.

The first external-domain pair is:

1. **ISIC 2019 skin-lesion classification**: multi-source medical dermoscopy, 25,331 labelled training images, 8 diagnostic classes. Use the official lesion identifier to prevent same-lesion leakage. The official dataset is CC-BY-NC; use it only for non-commercial academic research.
2. **Fish4Knowledge FRGT**: underwater fish-species classification, 27,370 images, 23 species. The original filenames encode tracking ID and fish ID; images from one tracking ID belong to the same trajectory and must stay in one split.

## Why the next run is a data gate, not immediate GPU training

The current PI-H2T review implementation in this repository is CIFAR-100-LT / ResNet-32 specific and has not been validated as a paper-faithful large-image implementation. The H2T repository has large-image dataset loaders, but not a complete ready-to-run external-domain recipe. Directly changing a dataset name and starting training would confound data adaptation, architecture adaptation, and method behavior.

This iteration therefore performs one small, CPU-only, reproducible data-readiness audit. It is not another internal H2T module ablation and it does not claim that either method fails.

## Locked split rules

### ISIC 2019

- Use only the official 25,331-image labelled training set and its training metadata.
- Use the official 8 one-hot diagnostic labels.
- Group by lesion ID; if it is missing, fall back to a unique image ID and record the fallback count.
- Exact duplicate image bytes are merged into the same leakage group before splitting.
- Per-class group split: 70% train / 15% validation / 15% test, with at least 5 independent groups in test, 3 in validation, and 5 in training.
- A class that cannot satisfy these minima is excluded from the primary manifest and explicitly reported. The dataset is NO-GO for the planned 8-class confirmatory screen unless all 8 classes qualify and the official image counts match.

### Fish4Knowledge FRGT

- Use the official image archives, not mask images.
- Parse class ID from the per-class archive folder (fish_01 or 01.Species) and parse each filename as tracking ID plus fish ID.
- All images sharing a tracking ID are one independent group. Exact duplicate image bytes are also merged into one group.
- Per-class group split: 60% train / 20% validation / 20% test.
- Primary eligibility requires at least 9 train, 3 validation, and 3 independent test trajectories (at least 15 trajectories total). This is intentionally stricter than a random image split.
- The full 23-class audit is retained. Classes that fail the independent-trajectory minimum remain in manifest_all.csv and are listed as excluded from manifest_primary.csv; they are not silently deleted.
- A limited-scope 12+-class primary subset is acceptable for an exploratory cross-domain screen, but it must be named by its actual number of species, never described as the full 23-class benchmark.

## Leakage and integrity checks

- Verify each image can be decoded and record its width, height, and SHA256.
- Check expected per-class counts against the official release.
- For Fish4Knowledge, check both per-class image counts and per-class trajectory counts against the official source.
- Merge records that share a lesion/track group or exact image hash before assigning splits.
- Fail if one merged group contains conflicting labels.
- Verify that no group ID or exact image hash appears in more than one split.
- Preserve raw image counts and independent group counts separately; image count is not a substitute for independent sample count.
- Build frequency groups as Head / Medium / Tail tertiles from primary training image counts. These groups are specific to each dataset and are not numerically identical to H2T paper's CIFAR Many / Medium / Few thresholds.

## What this iteration will decide

- If ISIC passes, it is the first medical external-domain training candidate.
- If Fish4Knowledge passes only as PASS_LIMITED_SCOPE, it is a second, restricted underwater candidate.
- If either dataset fails its integrity or independent-group gate, do not start training on that dataset. Report the exact reason and select another dataset or a different evaluation design.
- Do not change the minimum-group thresholds after seeing model accuracy; these are data-validity criteria, not model-performance filters.

## What is deliberately not included yet

- No H2T or PI-H2T training.
- No new loss, donor selection, fusion ratio, channel selection, or feature module.
- No ConvNeXt V2 / FAST backbone comparison.
- No artificial class rebalancing of validation or test data.
- No official test-set tuning.
- No claiming an H2T / PI-H2T failure from dataset statistics alone.

After the audit passes, the next iteration will implement one common large-image backbone and paper-form H2T / PI-H2T adaptations, with matched no-fusion controls. Backbone choice and exact Stage-I / Stage-II schedules will be fixed before looking at test results.
