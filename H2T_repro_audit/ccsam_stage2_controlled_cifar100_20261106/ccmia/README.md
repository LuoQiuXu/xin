# CC-MIA

CC-MIA (Cross-Class Margin Influence Audit) is a paired C2/C3 diagnostic experiment for the existing CC-SAM Stage-II implementation.

It is deliberately isolated from the original C0-C4 and TPR code. All scripts import the existing implementation without modifying it.

## Run order

From this directory, using the existing xiaoju Conda environment:

~~~powershell
python -B -m py_compile ccmia_core.py train_ccmia.py analyze_ccmia.py test_ccmia_core.py
python -B -m unittest -v test_ccmia_core
python -B -u train_ccmia.py --phase preflight
python -B -u train_ccmia.py --phase smoke
python -B -u train_ccmia.py --phase formal
python -B -u analyze_ccmia.py
~~~

Formal is blocked unless preflight and smoke reports both pass. The code refuses overwrite/resume.

See CCMIA_PROTOCOL.md for the exact math, frozen settings, interpretation limits and outputs. See CODEX_CCMIA_EXECUTE_20261117.txt for the full Codex task.

The 1,980-image validation set is a TRAIN-derived diagnostic holdout, not an independent test set. Official CIFAR-100 TEST is prohibited.
