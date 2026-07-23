# 声明—证据表 — 方向五：Grad-CAM 物理一致性量化

| # | 声明 (Claim) | 证据 (Evidence) | 证据文件 | 强度 | 风险 | 需修改 |
|---|-------------|----------------|---------|------|------|--------|
| C1 | STFT+2D-CNN在CWRU上达到极高准确率（3类） | 5种子均值99.86%，标准差<0.1% | results/experiment_results_v2.json cwru.accuracies | 强 | 3类任务太简单（含Normal） | 标注3类局限性 |
| C2 | STFT+2D-CNN在MFPT上达到100%准确率 | 5种子均100% | results/experiment_results_v2.json mfpt.accuracies | 强 | MFPT数据量小（1800段），文件少 | 标注样本量局限 |
| C3 | CWRU模型的Grad-CAM显著性未能集中在理论故障频带内 | PCS=1.76%，与平移频带的PCS（1.69%）无显著差异（p=0.52, d=0.05） | Figure cwru_pcs_box.png, cwru_pcs_acc_matrix.png | 强 | 频带定义依赖转速精确性 | 标注带宽敏感性局限 |
| C4 | MFPT模型的Grad-CAM对故障频率同样无显著关注 | PCS=1.21%，与平移频带（1.29%）无显著差异（p=0.11） | Figure mfpt_pcs_box.png | 中 | MFPT测试集仅4条录音，样本量小 | 标注样本量 |
| C5 | 双数据集均表现为"投机取巧模型"——高准确率+低物理一致性 | CWRU和MFPT的PCS均<2%，但准确率>99% | Figures cwru_pcs_acc_matrix.png, mfpt_pcs_acc_matrix.png | 强 | PCS定义和频带选择可能有替代方案 | 保留对PCS定义的有效性讨论 |
| C6 | 注入噪声反而提高CWRU上的PCS | SNR=10dB时PCS从1.76%升至3.23% | Figure cwru_noise_curve.png | 中 | 噪声在STFT幅值谱上注入，非时域 | 明确噪声注入方式 |
| C7 | MFPT模型在噪声下PCS崩溃（5dB→0.03%） | PCS从1.21%降至0.03%（SNR=5dB） | Figure mfpt_noise_curve.png | 中 | 同上 | 同上 |
| C8 | 跨数据集迁移完全失败 | CWRU训练模型在MFPT上准确率仅4.3%（低于随机33%） | results/experiment_results_v2.json cross_dataset | 强 | 归一化统计量来自CWRU | 已是实验的自然结果 |
| C9 | 深层Grad-CAM的PCS不优于浅层（CWRU层消融） | conv1 PCS=3.47%、conv2=3.42%、conv3=2.80% | Figure cwru_layer_pcs.png | 中 | 浅层PCS实际上更高 | 这本身是反直觉发现，值得报告 |
| C10 | MFPT层消融中PCS波动大但总体极低 | conv1=0.55%、conv2=1.46%、conv3=1.01%（标准差大） | Figure mfpt_layer_pcs.png | 弱 | MFPT测试集小，标准差大 | 降级声明强度 |

## 核心声明（论文中可以坚定陈述）

1. **两数据集均表现为"高准确率+低物理一致性"**（C1-C5, 强证据）
2. **跨数据集迁移完全失败**（C8, 强证据）——这是 CWRU 固有局限的实证
3. **噪声对PCS的影响不一致**（C6-C7, 中等证据）

## 过度声明（需要弱化措辞）

- "模型不使用故障频率信息" → 应改为 "在本实验的PCS定义和频带设置下，未观察到Grad-CAM显著性集中在理论故障频带的证据"
- "Grad-CAM无效" → 应改为 "标准Grad-CAM在本设置下未能提供物理一致的解释"

## 未支持声明（不应出现在论文中）

- "模型使用了投机取巧的特征" → 我们确实知道PCS很低，但不知道模型具体用了什么特征
- "其他故障诊断模型也有此问题" → 本实验仅测试了一种2D-CNN架构
