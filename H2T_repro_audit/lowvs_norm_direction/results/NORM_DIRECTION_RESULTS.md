# Low-VS vs Uniform: norm matching vs direction

Locked Clean StageI, fixed epoch10, 1980 reused TRAIN validation images (class0 absent).
Same input/VS/masks checked across four arms, 850 batches per seed. No official TEST.

|Seed|Arm|Overall|Head|Tail|Few|Avg alphaU|Avg alphaL|
|---|---|---:|---:|---:|---:|---:|---:|
|124|Uniform-Full|47.0707|60.1429|26.8966|27.5000|0.996|0.894|
|124|LowVS-Full|47.3232|59.8571|27.7586|28.3333|0.996|0.894|
|124|Uniform-Matched|47.0707|60.1429|26.8966|27.5000|0.996|0.894|
|124|LowVS-Matched|47.3232|60.1429|27.4138|28.0000|0.996|0.894|
|125|Uniform-Full|46.8687|60.1429|26.8966|27.5000|0.996|0.891|
|125|LowVS-Full|47.1717|59.5714|27.9310|28.5000|0.996|0.891|
|125|Uniform-Matched|46.8687|60.1429|26.8966|27.5000|0.996|0.891|
|125|LowVS-Matched|46.7172|60.0000|26.7241|27.3333|0.996|0.891|

## Paired changes, fixed epoch10
|Seed|Contrast|Overall pp|Tail pp|Head pp|Net images|
|---|---|---:|---:|---:|---:|
|124|LowVS-Full - Uniform-Full|+0.2525|+0.8621|-0.2857|+5|
|124|LowVS-Matched - Uniform-Matched|+0.2525|+0.5172|+0.0000|+5|
|124|LowVS-Matched - LowVS-Full|+0.0000|-0.3448|+0.2857|+0|
|124|Uniform-Matched - Uniform-Full|+0.0000|+0.0000|+0.0000|+0|
|125|LowVS-Full - Uniform-Full|+0.3030|+1.0345|-0.5714|+6|
|125|LowVS-Matched - Uniform-Matched|-0.1515|-0.1724|-0.1429|-3|
|125|LowVS-Matched - LowVS-Full|-0.4545|-1.2069|+0.4286|-9|
|125|Uniform-Matched - Uniform-Full|+0.0000|+0.0000|+0.0000|+0|

Read paired.csv and classifier_hybrids.csv before attributing mechanisms.
Key comparison: LowVS-Matched vs Uniform-Matched, next to the two Full arms.
Matched GAP norm does not match full-map L2, spatial structure, or gradient distributions.
If matched advantage persists, GAP norm alone is insufficient, NOT proof of useful semantics.
If matched advantage vanishes, norm is plausible, NOT uniquely established.
Do not sweep strengths or masks on this repeatedly-used dev validation split.
