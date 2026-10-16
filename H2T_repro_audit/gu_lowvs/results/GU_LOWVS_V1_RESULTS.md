# GU-LowVS V1 | primary epoch10, Clean TRAIN validation
StageI frozen, stageII seeds124/125; 99 classes, 1980 TRAIN holdout.
Official TEST not used. Clean reference gradient uses TRAIN only.
All paired per-batch indices/images/features/VS scores/Gumbel verified.

|Seed|Arm|Overall|Head|Tail|Few|Alpha0%|Alpha0.5%|Alpha1%|Best ep|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|124|LowVS-Full|47.323|59.857|27.759|28.333|0.0|0.0|100.0|9|
|124|LowVS-Half|47.071|61.571|26.207|26.667|0.0|100.0|0.0|9|
|124|GU-LowVS|46.515|63.571|23.793|24.167|54.2|14.0|31.8|7|
|124|GU-Shuffled|47.172|61.143|27.069|27.500|56.6|9.9|33.5|7|
|125|LowVS-Full|47.172|59.571|27.931|28.500|0.0|0.0|100.0|6|
|125|LowVS-Half|47.172|61.714|26.207|26.667|0.0|100.0|0.0|10|
|125|GU-LowVS|46.515|63.286|23.793|24.167|54.3|13.3|32.4|7|
|125|GU-Shuffled|46.818|60.857|26.552|27.000|55.4|10.1|34.5|8|

Exploratory method gate: NO-GO
Mean Overall vs full: -0.7323pp; vs shuffled: -0.4798pp
Read paired.csv + per_class.csv for actual wins/losses, not only means.
High clean TRAIN gradient alignment is NOT proof of unseen utility.
If GU loses to shuffled or almost always chooses alpha=0, stop this V1.
Do not tune candidate alphas, class weights, or reference bank on this validation set.
No additional StageI retraining and no claims about learned backbone features.
