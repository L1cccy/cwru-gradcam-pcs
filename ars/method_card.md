# 方法卡片 — 方向五：Grad-CAM 物理一致性量化

## 基本信息

| 项目 | 内容 |
|------|------|
| 方法名称 | Grad-CAM Physical Consistency Score (Grad-CAM PCS) |
| 解决的问题 | Grad-CAM 热力图常被定性展示为"模型可解释性证据"，但缺乏与物理参考频带的定量对比。高准确率 + 热区在非故障区 = 投机取巧模型（ch49） |
| 研究假设 | 若 STFT+2D-CNN 学到了故障频谱特征，则正确分类样本的 Grad-CAM 显著性在理论故障频带（BPFO/BPFI/BSF）内的占比应显著高于平移频带（负对照） |

## 方法详述

### 输入格式与维度

| 参数 | 值 |
|------|-----|
| 原始信号窗口长度 | 1024 点（~85 ms @ 12 kHz） |
| STFT: n_fft | 512（频率分辨率 23.44 Hz，可区分 BPFI≈157Hz 与 BSF≈137Hz） |
| STFT: hop_length | 128（生成约 5 个时间帧） |
| STFT: window | Hann |
| 2D-CNN 输入形状 | [1, 257, 5] = [ch, freq_bins, time_frames] |

### 模型架构

```
2D-CNN（ch41 标准架构）:
  Conv2D(1, 16, kernel=3, padding=1) → BN → ReLU → MaxPool(2)
  Conv2D(16, 32, kernel=3, padding=1) → BN → ReLU → MaxPool(2)
  Conv2D(32, 64, kernel=3, padding=1) → BN → ReLU → AdaptiveAvgPool(1)
  Flatten → Linear(64, 128) → ReLU → Dropout(0.5) → Linear(128, 4)
```

输出 4 类：Normal / Ball / InnerRace / OuterRace

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam, lr=0.001 |
| 训练轮次 | 50 |
| Batch size | 32 |
| 损失函数 | CrossEntropyLoss |
| 随机种子 | 42, 123, 256（3 个种子） |
| 划分方式 | 按 recording_id 分组（80% train / 20% test），同记录的窗口不在两个划分中 |
| Early stopping | patience=10, monitor val_loss |

### Grad-CAM 生成流程

```
1. 前向传播 → 获取目标类 logit + target_conv_layer 特征图 A^k (shape: [C, H, W])
2. 反向传播 → ∂y_c / ∂A^k 梯度 → 全局平均池化 → 权重 α_k^c
3. L_Grad-CAM = ReLU(∑ α_k^c · A^k)  → 上采样至输入尺寸 [257, 5]
4. 沿时间轴平均 → 频率显著性曲线 S(f) (长度 257)
5. PCS = ∑_{f ∈ fault_bands} S(f) / ∑_{f} S(f)
```

### 故障频率参考频带

基于 CWRU SKF 6205-2RS 轴承参数（节径 39.04mm, 滚珠直径 7.94mm, 滚珠数 9, 接触角 0°）：

| 故障类型 | 特征频率 | 参考频带（±2 bins） |
|---------|---------|-------------------|
| BPFO | 3.05 × f_r ≈ 104 Hz | 70–140 Hz |
| BPFI | 4.95 × f_r ≈ 157 Hz | 117–188 Hz |
| BSF | 2.00 × f_r ≈ 137 Hz | 94–164 Hz |
| 2× BPFO | 208 Hz | 164–234 Hz |
| 3× BPFO | 312 Hz | 281–352 Hz |

实际值随转速（1772–1797 rpm）略有偏移，在实验中精确计算。

## 基线设计

| 基线 | 类型 | 描述 |
|------|------|------|
| Random Guessing | Simple | 4 类随机猜测基准（25%） |
| STFT+2D-CNN（无 Grad-CAM 分析） | Standard | ch41 标准分类器 |
| 频带平移负对照 | Negative Control | 将故障频带沿频率轴随机平移，比较 PCS_true vs PCS_shifted |

## 机理验证方案（分类前完成）

这是本方向的核心——机理验证即主体方法：

1. **计算理论故障频率**：从轴承几何参数 + 转速精确计算 BPFO/BPFI/BSF
2. **Grad-CAM 生成**：对正确分类的测试样本生成热力图
3. **PCS 计算**：计算故障频带内显著性占比
4. **负对照设计**：频带平移后重算 PCS，配对检验
5. **三种可能结果预判**：

| 结果 | 判断 |
|------|------|
| PCS_true >> PCS_shifted, p < 0.01 | 正面：模型确实关注故障频域特征 |
| PCS_true ≈ PCS_shifted, p > 0.05 | 零结果：热力图不能支持解释，模型可能是投机取巧 |
| PCS_true < PCS_shifted | 负面：模型系统性地回避故障频带 |

## 消融方案

| 消融 | 操作 | 回答的问题 |
|------|------|-----------|
| A: 浅层 vs 深层 | 分别对 conv1, conv2, conv3 生成 Grad-CAM，比较 PCS | 不同层的物理一致性是否有差异？深层是否更关注故障频率？ |
| B: Grad-CAM vs Grad-CAM++ | 对同一批样本用 Grad-CAM++ 生成热力图，比较 PCS 和跨种子稳定性 | 不同解释方法的结论是否一致？ |

## 非精度维度

- **物理一致性 × 准确率 2×2 矩阵**（高/low PCS × 高/low Accuracy）→ 部署决策
- **跨种子 PCS 稳定性**：3 种子的 PCS 均值 ± 标准差
- **跨窗口稳定性**：同一记录中相邻 5 个窗口的 Grad-CAM Jaccard 相似度
- **跨噪声鲁棒性**：SNR=10dB, 5dB 噪声下的 PCS 退化曲线

## 实验矩阵

| 实验ID | 模型 | 划分 | 种子 | Grad-CAM目标层 | 测试条件 | 主指标 | 非精度指标 |
|--------|------|------|------|---------------|---------|--------|----------|
| E1 | 2D-CNN | rec_id | 42 | - | 干净 | Accuracy | Per-class recall |
| E2 | 2D-CNN | rec_id | 123 | - | 干净 | Accuracy | Per-class recall |
| E3 | 2D-CNN | rec_id | 256 | - | 干净 | Accuracy | Per-class recall |
| E4 | 2D-CNN | rec_id | 42 | conv3 | 干净 | PCS_true | PCS_shifted, ΔPCS |
| E5 | 2D-CNN | rec_id | 123 | conv3 | 干净 | PCS_true | 跨种子 Jaccard |
| E6 | 2D-CNN | rec_id | 256 | conv3 | 干净 | PCS_true | 跨种子 Jaccard |
| E7 | 2D-CNN | rec_id | 42 | conv3 | SNR=10dB | PCS_noisy | ΔPCS vs 干净 |
| E8 | 2D-CNN | rec_id | 42 | conv3 | SNR=5dB | PCS_noisy | ΔPCS vs 干净 |
| E9 | 2D-CNN | rec_id | 42 | conv1,2,3 | 干净 | PCS_per_layer | 逐层 PCS |
| E10 | 2D-CNN | rec_id | 42 | conv3 (Grad-CAM++) | 干净 | PCS_plus | ΔPCS vs Grad-CAM |

## 预期失败情况

- STFT 频率分辨率不足，BPFI 和 BSF 频带重叠 → 改用更大 n_fft
- 由于 2D-CNN 在 CWRU 上准确率过高（~95%+），错误分类样本过少 → 报告对此局限
- Grad-CAM 的空间分辨率受限于最后一层 conv 的特征图尺寸 → 标注此局限

## 主要风险

1. 故障频率计算依赖准确转速，CWRU 的实际转速与标称值可能有偏差
2. STFT 频率分辨率（23.44 Hz）仍然较粗，某些谐波可能跨 bin
3. Grad-CAM 热力图沿时间轴平均可能丢失动态信息
