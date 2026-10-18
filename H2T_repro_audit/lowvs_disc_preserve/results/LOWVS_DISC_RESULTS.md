# Low-VS + clean discriminative preservation: three-arm pilot

Primary contrast: TopCompetitor vs CleanCE, NOT merely vs LowVS.
Historical LowVS Full signature 850/850 checked in each arm and each seed.
See group_confusion.csv for true group versus predicted group, not exact-label accuracy.
See historical_replay.csv for numerical drift from prior sealed LowVS-Full.
Frozen StageI seed42 epoch184; StageII fixed epoch10; 1980 TRAIN holdout class1..99 (20/class); no official TEST.

| Seed | Arm | Overall | Head | OfficialMedium | Tail | Few | validation CE |
|---|---|---:|---:|---:|---:|---:|---:|
|124|LowVS|47.3232|59.8571|51.0000|27.7586|28.3333|2.1530|
|124|LowVS-CleanCE|46.7172|58.8571|48.7143|29.6552|30.1667|2.1632|
|124|LowVS-TopCompetitor|46.9192|59.1429|49.4286|29.1379|29.6667|2.1564|
|125|LowVS|47.1717|59.5714|50.7143|27.9310|28.5000|2.1532|
|125|LowVS-CleanCE|46.5152|58.0000|48.8571|29.8276|30.3333|2.1645|
|125|LowVS-TopCompetitor|46.6667|58.7143|49.4286|28.7931|29.3333|2.1572|

|Seed|Contrast|Overall pp|Head pp|Tail pp|Net images|Tail net|Tail→Head change|
|---|---|---:|---:|---:|---:|---:|---:|
|124|LowVS-CleanCE - LowVS|-0.6061|-1.0000|+1.8966|-12|+11|-20|
|124|LowVS-TopCompetitor - LowVS|-0.4040|-0.7143|+1.3793|-8|+8|-12|
|124|LowVS-TopCompetitor - LowVS-CleanCE|+0.2020|+0.2857|-0.5172|+4|-3|+8|
|125|LowVS-CleanCE - LowVS|-0.6566|-1.5714|+1.8966|-13|+11|-13|
|125|LowVS-TopCompetitor - LowVS|-0.5051|-0.8571|+0.8621|-10|+5|-10|
|125|LowVS-TopCompetitor - LowVS-CleanCE|+0.1515|+0.7143|-1.0345|+3|-6|+3|

**Interpretation precommitted**:
- If TopCompetitor beats CleanCE on Tail in both seeds without consequential Overall/Head loss, targeted clean discrimination earns follow-up consideration, not immediate novelty.
- If both auxiliary arms behave alike, added clean-label supervision is a sufficient explanation: do not claim a special competitor mechanism.
- If both harm LowVS, adding clean constraints is not justified on this frozen FC; stop this module pilot.
- If results change sign by seed or net few images, do not tune lambda or margins on this reused dev split.
- Auxiliary CE and pairwise softplus have different gradient magnitudes; same lambda alone does not equate gradient budget.
- The classifier alone is trained: cannot claim Backbone feature learning or publication-grade generalization.