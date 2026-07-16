# 实验协议 — 方向五：Grad-CAM 物理一致性量化

## §9 机理验证

> manual §9: "Do not jump straight to classification. First show that the method changes the representation in the intended way."

方向五的特殊性在于：**机理验证即主体方法本身**。我们要验证的是"模型是否真正关注了故障频域特征"，而不是"STFT 是否表示了故障信息"。

### 机理测试 1：故障频带内显著性占比（Physical Consistency Score, PCS）

**测量对象**: 正确分类测试样本的 Grad-CAM 热力图

**具体步骤**:
1. 训练 STFT+2D-CNN 分类器（recording-level split, seed=42）
2. 对每个正确分类的测试样本，在最后一层卷积层（conv3）生成 Grad-CAM 热力图
3. 热力图沿时间轴平均 → 频率显著性曲线 S(f)，长度 = n_fft/2+1 = 257
4. 计算故障频带（BPFO/BPFI/BSF ± 2 bins）内的 S(f) 能量和 / 全频带 S(f) 能量和 = PCS
5. 对全部正确分类样本取均值 → 模型级 PCS

**频带定义**（基于 CWRU SKF 6205-2RS，~1750 rpm）:

| 频带 ID | 频率范围 (Hz) | 对应的 bin 范围 (n_fft=512) | 理论来源 |
|---------|-------------|--------------------------|---------|
| B1 | 82–129 | 4–6 | BPFO 基频（~104 Hz） |
| B2 | 129–175 | 6–8 | BSF（~137 Hz）邻近 |
| B3 | 129–188 | 6–8 | BPFI 基频（~157 Hz） |
| B4 | 188–234 | 8–10 | 2×BPFO（~208 Hz） |
| B5 | 281–328 | 12–14 | 3×BPFO（~312 Hz） |

**预期图表**: 柱状图——每个故障类别的 PCS 均值 ± 标准差（3 种子）

**支持方法的证据**: PCS > 0.3（30% 以上显著性落在故障频带内），且各类别的 PCS 与对应故障频率一致

**削弱方法的证据**: PCS < 0.1，或各类别 PCS 无差异，或标准差极大

### 机理测试 2：频带平移负对照（负对照的核心）

**测量对象**: PCS_true vs PCS_shifted

**具体步骤**:
1. 生成 5 组随机平移频带（保持相同宽度，随机偏移 ±50~100 Hz）
2. 在平移频带上计算 PCS_shifted（共 5 组）
3. 对 PCS_true 和 PCS_shifted 做配对 t 检验或计算 Cohen's d 效应量

**预期图表**: 箱线图——PCS_true vs PCS_shifted 的分布对比，标注 p 值和效应量

**支持方法的证据**: PCS_true 显著高于 PCS_shifted（p < 0.05, Cohen's d > 0.8）

**证伪（机制失败）**: PCS_true 与 PCS_shifted 无显著差异（p > 0.05）

### 机理测试 3：正确分类 vs 错误分类的 PCS 对比

**测量对象**: 正确分类样本的 PCS vs 错误分类样本的 PCS

**预期**: 错误分类样本的 PCS 应显著低于正确分类样本（如果分类失败意味着模型没有正确关注故障频域）

**可能的混淆因素**: 错误分类样本数可能很少（2D-CNN 在 CWRU 上准确率 >95%），样本量不足

---

## §10 分类训练协议

> manual §10: "Fix: segment length, sampling rate, preprocessing, split, architecture, optimizer, lr, batch size, epochs, early stopping, random seeds."

### 10.1 固定参数

| 参数 | 值 |
|------|-----|
| 窗口长度 | 1024 点 |
| 采样率 | 12 kHz（Normal数据从48kHz降采样至12kHz） |
| STFT: n_fft | 512 |
| STFT: hop_length | 128 |
| STFT: window | Hann |
| 输入尺寸 | [1, 257, 5] |
| 训练/测试比例 | 80% / 20%（按 recording_id 分组分层采样） |
| 模型 | 2D-CNN (Conv2D [16,32,64]) |
| 优化器 | Adam, lr=0.001 |
| Batch size | 32 |
| Epochs | 50 |
| Early stopping | patience=10, monitor=val_loss |
| 随机种子 | 42, 123, 256 |

### 10.2 报告指标

| 指标 | 计算 |
|------|------|
| Accuracy | 整体准确率（3 种子均值 ± 标准差） |
| Macro-F1 | 宏平均 F1 |
| Per-class precision/recall | 每类精度和召回率 |
| Confusion matrix | 4×4 混淆矩阵（累计 3 种子） |

### 10.3 预期结果表格式

| 种子 | Accuracy | Macro-F1 | B_recall | IR_recall | OR_recall | N_recall |
|------|----------|----------|----------|-----------|-----------|----------|
| 42 | | | | | | |
| 123 | | | | | | |
| 256 | | | | | | |
| Mean±SD | | | | | | |

---

## §11 物理一致性（非精度维度）

> manual §11.3: "Check: fault-frequency band energy, envelope spectrum peaks, saliency near physically meaningful regions."

方向五的非精度维度即为物理一致性本身，与 §9 机理验证共用指标。

### 11.1 核心表格：物理一致性 × 准确率 2×2 矩阵

| | 高准确率 (>90%) | 低准确率 |
|---|---|---|
| **高 PCS (>0.3)** | ✅ 理想：模型准确 + 关注故障频率 | ❓ 模型关注频率但判错（标签错误？） |
| **低 PCS (<0.15)** | ⚠️ 投机取巧：准确但关注非故障频率 | ❌ 差模型：既不准确也无关注意义 |

### 11.2 跨种子稳定性

- 3 种子模型的 PCS 均值 ± 标准差
- 每对种子间的 Grad-CAM Jaccard 相似度（SSIM 作为补充）
- 预期图：3 种子的频率显著性曲线叠加图

### 11.3 跨窗口稳定性

- 同一 recording_id 中取 5 个连续窗口
- 计算 5 个窗口的 Grad-CAM 互相关或 Jaccard 相似度

### 11.4 噪声鲁棒性

- 测试集加噪：SNR=10dB, 5dB（高斯白噪声）
- 测量 PCS 随 SNR 的退化曲线
- 对比 PCS 退化速度与 Accuracy 退化速度

---

## §12 消融

> manual §12: "Remove or alter one component at a time. Ask: Which part matters?"

### 消融 A：浅层 vs 深层 Grad-CAM

| 变体 | 目标层 | 要回答的问题 |
|------|--------|------------|
| A1 | conv1（浅层） | 浅层特征图是否也保留频域信息？ |
| A2 | conv2（中层） | 中间层的频率选择性 |
| A3 | conv3（深层）| 主实验（已包含在 §9 中） |

**预期结果**: 深层 PCS > 浅层 PCS（如果卷积层级联确实提高了频率选择性）

### 消融 B：Grad-CAM vs Grad-CAM++

| 变体 | 方法 | 要回答的问题 |
|------|------|------------|
| B1 | Grad-CAM（标准） | 主实验 |
| B2 | Grad-CAM++ | 加权平均是否能产生更稳定的解释？ |

**预期结果**: Grad-CAM++ 的跨种子 Jaccard 相似度 > Grad-CAM

### 消融最小表格格式

| 消融 | 变体 | PCS_true | PCS_shifted | ΔPCS | Cohen's d |
|------|------|----------|-------------|------|-----------|
| A | conv1 (shallow) | | | | |
| A | conv2 (mid) | | | | |
| A | conv3 (deep) | | | | |
| B | Grad-CAM | | | | |
| B | Grad-CAM++ | | | | |
