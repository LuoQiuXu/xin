# H2T IF200 controlled stress | fixed-BN, fixed-epoch results

IF200 nested 9502 LT TRAIN images; original 1980-image TRAIN holdout (class0 absent).
StageI fresh seed42, 200 epochs fixed checkpoint. StageII 10epochs, seeds124/125; single frozen StageI.
Four arms: DR-NoFusion, Original-BatchShared, Uniform-PerSample, LowVS-PerSample.
Original denotes the mask-and-sampling analogue; NOT author's shift_bn=True full implementation.

|Seed|Arm|Overall|Head|Tail|WorstClass|
|---|---|---:|---:|---:|---:|
|124|DR-NoFusion|41.616|63.286|16.724|0.000|
|124|Original-BatchShared|42.828|62.714|17.931|0.000|
|124|Uniform-PerSample|42.727|62.857|17.759|0.000|
|124|LowVS-PerSample|42.424|61.286|18.103|0.000|
|125|DR-NoFusion|41.869|63.714|16.897|0.000|
|125|Original-BatchShared|42.626|62.429|17.931|0.000|
|125|Uniform-PerSample|42.727|62.714|18.276|0.000|
|125|LowVS-PerSample|42.222|61.000|18.103|0.000|

## Paired changes, fixed epoch10
|Seed|A - B|Wins|Losses|Net|Head net|Tail net|Tail true→Head predicted delta|
|---|---|---:|---:|---:|---:|---:|---:|
|124|Original-BatchShared - DR-NoFusion|56|32|+24|-4|+7|-46|
|124|Uniform-PerSample - Original-BatchShared|7|9|-2|+1|-1|+2|
|124|LowVS-PerSample - Uniform-PerSample|13|19|-6|-11|+2|-9|
|124|LowVS-PerSample - Original-BatchShared|12|20|-8|-10|+1|-7|
|125|Original-BatchShared - DR-NoFusion|47|32|+15|-9|+6|-42|
|125|Uniform-PerSample - Original-BatchShared|10|8|+2|+2|+2|-1|
|125|LowVS-PerSample - Uniform-PerSample|10|20|-10|-12|-1|-13|
|125|LowVS-PerSample - Original-BatchShared|17|25|-8|-10|+1|-14|

### Decision and limits
If OriginalShared consistently harms Tail vs DR, this is an exploratory IF200 negative-transfer candidate, not proof of dataset-invariant failure.
If LowVS improves only against Uniform and not Original, do not claim a meaningful original H2T replacement.
Do not infer IF interaction without a harmonized IF100 within-setting replication (historical data are protocol-distinct).
No alpha/K/hardness/optimizer tuning. Compare per_class.csv and paired.csv before interpreting worst-class zeros.
No public CIFAR100 TEST was used; do not call the reused 99-class TRAIN holdout an independent final test.