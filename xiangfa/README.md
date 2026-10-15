# 反事实类别支撑动力学实验

本实验验证两个相互区分的问题：

1. 当同一语义类别经历不同训练频率时，其类别支撑区域是否发生稳定、具有剂量关系的变化。
2. 从其他类别学到的频率—支撑变化，能否在类别不相交设置下恢复未知尾类的充分样本支撑。

## 严格控制

- 使用 5 个频率等级：`5, 20, 50, 200, 500`。
- 构造 5 个环境，每个环境都有完全相同的频率直方图。
- 每个类别在 5 个环境中恰好经历每个频率一次。
- 每类的小样本集合严格嵌套在大样本集合中。
- 机制分析始终使用同一批平衡测试图像，避免把估计样本量差异误认为表征变化。
- 恢复器使用类别不相交五折验证。

## 支撑描述

支撑区域不只使用类别原型，而是联合描述：

- 多分位角半径：50%、80%、90%、95%；
- 切空间协方差迹、有效秩和前 8 个特征值；
- 最近邻类别中心关系；
- 样本余弦边际和重叠率；
- 超球面帽支撑区域的重叠比例和重叠程度。

## 两类恢复任务

- `fixed_eval_dynamics`：输入和目标都来自固定平衡测试集，用于隔离模型学习造成的真实支撑变形。
- `fewshot_to_head`：输入只由尾类实际可见的少量训练样本估计，目标为充分样本状态下的固定测试支撑，更接近真实应用。

## 对照模型

- `identity`：不恢复；
- `count_only`：只学习每个频率对的平均退化量；
- `prototype_ridge`：仅使用类别中心邻域关系；
- `cap_ridge`：使用“原型关系 + 单一 95% 角半径”，对应超球面帽的简化描述；
- `full_support_no_count_ridge`：使用完整支撑描述但不输入频率，用于排除普通 train-test 校准；
- `full_support_ridge`：使用完整支撑描述；
- `full_support_extra_trees`：完整支撑描述的非线性探索模型。

## 运行

使用已配置的 PyTorch 环境：

```powershell
D:\app\anaconda\envs\pytorch\python.exe support_dynamics_experiment.py --smoke
```

正式实验：

```powershell
D:\app\anaconda\envs\pytorch\python.exe support_dynamics_experiment.py --epochs 80 --seeds 42
```

复用已有检查点重新分析：

```powershell
D:\app\anaconda\envs\pytorch\python.exe support_dynamics_experiment.py --epochs 80 --seeds 42 --skip-train
```

结果默认保存在 `outputs/support_dynamics/`。

## 平衡空间邻居—头尾轨迹一致性实验

`neighbor_trajectory_experiment.py` 先从原始 CIFAR-100 训练集中构造独立的平衡参考划分：

- 每类 400 张训练平衡参考模型；
- 每类另外 100 张计算类别原型、对称混淆和局部 KNN 重叠；
- 三项信号使用 reciprocal-rank fusion 预先确定 Top-k 相似类。

相似类确定后，程序读取五频率受控实验已经生成的
`support_states.csv`，比较这些预选类别对在
`500→200→50→20→5` 过程中的类内支撑增量轨迹。轨迹评价来自原始
CIFAR-100 测试集，与筛选相似类的平衡训练/验证划分完全分离。

冒烟测试：

```powershell
D:\app\anaconda\envs\pytorch\python.exe neighbor_trajectory_experiment.py --smoke
```

正式实验：

```powershell
D:\app\anaconda\envs\pytorch\python.exe neighbor_trajectory_experiment.py --epochs 100 --seeds 42
```
