# Low-VS 边界方向机制审计（TRAIN-only；无需训练）

固定Clean Stage-I分类器的竞争类别 j*，不使用验证/TEST选择类别、掩码、参数或阈值。
主指标是尾类有效跨类融合的归一化方向投影 qL-qU；更负只代表更强的即时判别干扰，不代表泛化改善。
两个StageII seed在同一个StageI骨干上产生不同增强视图，不是独立训练模型。

|Seed|Group|Draws|mean(qL-qU)|P(qL<qU)|mean(cosL-cosU)|matched Δmargin L-U|full Δmargin L-U|extra magnitude term|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|124|AllEligible|10386|-0.045981|0.580|-0.016251|-0.144231|-0.263628|-0.119397|
|124|TailEligible|3064|-0.037316|0.595|-0.014452|-0.121910|-0.221270|-0.099359|
|124|TailEligibleDonorHead|2603|-0.037626|0.596|-0.014713|-0.121392|-0.221862|-0.100470|
|124|HeadEligible|3640|-0.050071|0.566|-0.016644|-0.148502|-0.261573|-0.113071|
|125|AllEligible|10378|-0.045276|0.577|-0.015941|-0.142061|-0.253372|-0.111311|
|125|TailEligible|2986|-0.040377|0.586|-0.015458|-0.128256|-0.228908|-0.100652|
|125|TailEligibleDonorHead|2523|-0.042096|0.589|-0.016205|-0.133080|-0.233991|-0.100911|
|125|HeadEligible|3768|-0.047680|0.566|-0.015855|-0.141412|-0.248552|-0.107140|

**Mandatory scientific reading:**
q_yj<0 means the chosen direction compresses the existing true-vs-fixed-competitor margin. It can provide a difficult training example OR destructive label noise; this audit cannot decide which.
Identity: Full margin contrast = matched-at-r direction contribution + extra-magnitude contribution. The identity describes a frozen instant-margin, NOT final accuracy percentages.
Examine both TailEligible and TailEligibleDonorHead before proposing an algorithm. If full-margin contrast mostly comes from extra magnitude and matched direction contrast is tiny/inconsistent, do not claim VS provides useful boundary direction.
Original 1980-image TRAIN development set has already been reused to choose prior methods; no independent inference or significance claims.
Any subsequent transfer assessment must compare DR/Original-H2T/LowVS at same dataset, imbalance, Stage-I, preprocessing and training schedule, measuring negative transfer and head-tail impact.
