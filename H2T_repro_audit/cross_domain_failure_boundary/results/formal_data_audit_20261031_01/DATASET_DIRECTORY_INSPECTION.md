# DATASET_DIRECTORY_INSPECTION

检查日期：2026-10-31（Asia/Shanghai）。

数据根目录：`D:\xiangmuxunlian\xin\data`。使用 os.walk 递归全部子目录（含隐藏项），按大小写不敏感的 .csv 扩展名筛选。
共检查 95,667 个文件条目；遍历错误 0；目录链接 0。
本次只读检查目录、文件名和 CSV 大小，没有移动、删除、改写或下载数据。

## 全部 CSV

| 绝对路径 | 字节数 |
|---|---:|
| `D:\xiangmuxunlian\xin\data\ISIC_2019_Training_GroundTruth.csv` | 1291479 |
| `D:\xiangmuxunlian\xin\data\ISIC_2019_Training_Metadata.csv` | 1214351 |
| `D:\xiangmuxunlian\xin\data\class_id.csv` | 576325 |
| `D:\xiangmuxunlian\xin\data\OfficeHomeDataset_10072016\ImageInfo.csv` | 2673777 |

## ISIC 必需 CSV

- `ISIC_2019_Training_Metadata.csv`：`D:\xiangmuxunlian\xin\data\ISIC_2019_Training_Metadata.csv`。
- `ISIC_2019_Training_GroundTruth.csv`：`D:\xiangmuxunlian\xin\data\ISIC_2019_Training_GroundTruth.csv`。

其他数据集的 CSV 不能替代 ISIC 标签或 lesion_id 元数据。

## Fish4Knowledge 完整目录结构

下表列出该目录本身及全部子目录；路径相对于数据根目录。图像与 mask 分开计数，mask 不作为分类图像输入。

| 相对目录 | 直接图像文件数 | 扩展名及文件数 | 文件名样例 |
|---|---:|---|---|
| `fishRecognition_GT` | 0 | — | `—` |
| `fishRecognition_GT/fish_image` | 0 | — | `—` |
| `fishRecognition_GT/fish_image/fish_01` | 12112 | .png=12112 | `fish_000000009598_05281.png` |
| `fishRecognition_GT/fish_image/fish_02` | 2683 | .png=2683 | `fish_000000249596_03656.png` |
| `fishRecognition_GT/fish_image/fish_03` | 3593 | .png=3593 | `fish_000000679596_03664.png` |
| `fishRecognition_GT/fish_image/fish_04` | 4049 | .png=4049 | `fish_000000019599_07753.png` |
| `fishRecognition_GT/fish_image/fish_05` | 2534 | .png=2534 | `fish_003389985224_25846.png` |
| `fishRecognition_GT/fish_image/fish_06` | 190 | .png=190 | `fish_000004210001_00013.png` |
| `fishRecognition_GT/fish_image/fish_07` | 450 | .png=450 | `fish_000000019596_03647.png` |
| `fishRecognition_GT/fish_image/fish_08` | 218 | .png=218 | `fish_000005919596_03822.png` |
| `fishRecognition_GT/fish_image/fish_09` | 241 | .png=241 | `fish_000027420001_02911.png` |
| `fishRecognition_GT/fish_image/fish_10` | 299 | .png=299 | `fish_003478665350_12015.png` |
| `fishRecognition_GT/fish_image/fish_11` | 98 | .png=98 | `fish_000001329595_03643.png` |
| `fishRecognition_GT/fish_image/fish_12` | 147 | .png=147 | `fish_000020830001_01788.png` |
| `fishRecognition_GT/fish_image/fish_13` | 181 | .png=181 | `fish_000000999597_04992.png` |
| `fishRecognition_GT/fish_image/fish_14` | 90 | .png=90 | `fish_000003219596_03737.png` |
| `fishRecognition_GT/fish_image/fish_15` | 42 | .png=42 | `fish_000021980001_01920.png` |
| `fishRecognition_GT/fish_image/fish_16` | 206 | .png=206 | `fish_000000819594_03402.png` |
| `fishRecognition_GT/fish_image/fish_17` | 49 | .png=49 | `fish_000023150001_02181.png` |
| `fishRecognition_GT/fish_image/fish_18` | 56 | .png=56 | `fish_000027080001_02813.png` |
| `fishRecognition_GT/fish_image/fish_19` | 29 | .png=29 | `fish_003470035337_13984.png` |
| `fishRecognition_GT/fish_image/fish_20` | 21 | .png=21 | `fish_000024219597_05199.png` |
| `fishRecognition_GT/fish_image/fish_21` | 16 | .png=16 | `fish_000023840001_02333.png` |
| `fishRecognition_GT/fish_image/fish_22` | 41 | .png=41 | `fish_000021760001_01811.png` |
| `fishRecognition_GT/fish_image/fish_23` | 25 | .png=25 | `fish_000007060001_00789.png` |
| `fishRecognition_GT/mask_image` | 0 | — | `—` |
| `fishRecognition_GT/mask_image/mask_01` | 12112 | .png=12112 | `mask_000000009598_05281.png` |
| `fishRecognition_GT/mask_image/mask_02` | 2683 | .png=2683 | `mask_000000249596_03656.png` |
| `fishRecognition_GT/mask_image/mask_03` | 3593 | .png=3593 | `mask_000000679596_03664.png` |
| `fishRecognition_GT/mask_image/mask_04` | 4049 | .png=4049 | `mask_000000019599_07753.png` |
| `fishRecognition_GT/mask_image/mask_05` | 2534 | .png=2534 | `mask_003389985224_25846.png` |
| `fishRecognition_GT/mask_image/mask_06` | 190 | .png=190 | `mask_000004210001_00013.png` |
| `fishRecognition_GT/mask_image/mask_07` | 450 | .png=450 | `mask_000000019596_03647.png` |
| `fishRecognition_GT/mask_image/mask_08` | 218 | .png=218 | `mask_000005919596_03822.png` |
| `fishRecognition_GT/mask_image/mask_09` | 241 | .png=241 | `mask_000027420001_02911.png` |
| `fishRecognition_GT/mask_image/mask_10` | 299 | .png=299 | `mask_003478665350_12015.png` |
| `fishRecognition_GT/mask_image/mask_11` | 98 | .png=98 | `mask_000001329595_03643.png` |
| `fishRecognition_GT/mask_image/mask_12` | 147 | .png=147 | `mask_000020830001_01788.png` |
| `fishRecognition_GT/mask_image/mask_13` | 181 | .png=181 | `mask_000000999597_04992.png` |
| `fishRecognition_GT/mask_image/mask_14` | 90 | .png=90 | `mask_000003219596_03737.png` |
| `fishRecognition_GT/mask_image/mask_15` | 42 | .png=42 | `mask_000021980001_01920.png` |
| `fishRecognition_GT/mask_image/mask_16` | 206 | .png=206 | `mask_000000819594_03402.png` |
| `fishRecognition_GT/mask_image/mask_17` | 49 | .png=49 | `mask_000023150001_02181.png` |
| `fishRecognition_GT/mask_image/mask_18` | 56 | .png=56 | `mask_000027080001_02813.png` |
| `fishRecognition_GT/mask_image/mask_19` | 29 | .png=29 | `mask_003470035337_13984.png` |
| `fishRecognition_GT/mask_image/mask_20` | 21 | .png=21 | `mask_000024219597_05199.png` |
| `fishRecognition_GT/mask_image/mask_21` | 16 | .png=16 | `mask_000023840001_02333.png` |
| `fishRecognition_GT/mask_image/mask_22` | 41 | .png=41 | `mask_000021760001_01811.png` |
| `fishRecognition_GT/mask_image/mask_23` | 25 | .png=25 | `mask_000007060001_00789.png` |

## ISIC 图像目录

| 绝对目录 | 直接图像文件数 | 扩展名及文件数 | 图像文件名样例 |
|---|---:|---|---|
| `D:\xiangmuxunlian\xin\data\ISIC_2019_Training_Input` | 0 | — | `—` |
| `D:\xiangmuxunlian\xin\data\ISIC_2019_Training_Input\ISIC_2019_Training_Input` | 25331 | .jpg=25331, .txt=2 | `ISIC_0000000.jpg` |

目录检查仅确认数量、路径与扩展名；不将扩展名或文件数视为解码、八类标签、病灶独立性或无泄漏的证明。
尚未提供的必需 CSV 会阻止 ISIC 完整审计；没有据此启动训练。
