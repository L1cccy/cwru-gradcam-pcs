# 文献矩阵 — 方向五：Grad-CAM 物理一致性量化

## 文献概览

| # | 作者/年份 | 标题 | 来源 | 引用数 | 与我们研究的关系 |
|---|----------|------|------|--------|---------------|
| 1 | Chen & Lee (2020) | Vibration signals analysis by explainable AI (XAI) approach: Application on bearing faults diagnosis | IEEE Access | 225 | 首次在振动信号中应用 Grad-CAM，显示频域注意力 |
| 2 | Mey & Neufeld (2022) | Explainable AI algorithms for vibration data-based fault detection | Sensors | 63 | 明确分析频带内的重要性，提出了评估 XAI 方法的框架 |
| 3 | Li et al. (2023) | Multilayer Grad-CAM: An effective tool towards explainable deep neural networks for intelligent fault diagnosis | J. Manufacturing Systems | 141 | 定义了 3 个量化指标（RATM, RATA, CEI），最接近我们的量化思路 |
| 4 | Lu et al. (2023) | An interpretable deep learning method for bearing fault diagnosis | arXiv | 4 | 构建特征图健康库，展示物理有意义的 Grad-CAM |
| 5 | Guo et al. (2023) | An analysis method for interpretability of CNN in bearing fault diagnosis | IEEE Trans. Instrum. Meas. | 46 | 从时频域分析 CNN 可解释性 |
| 6 | Liefstingh et al. (2021) | Interpretation of deep learning models in bearing fault diagnosis | PHM Conf. | 21 | 系统分析 Grad-CAM 在轴承诊断中的解释效果 |
| 7 | Chen et al. (2024) | Enhancing reliability through interpretability: A comprehensive survey | IEEE Access | 76 | 综述文章，确认了现有工作未做负对照评估 |
| 8 | Kim & Kim (2024) | Vibration spectrogram analysis for bearing fault diagnosis based on Grad-CAM | JMST | 17 | 用 Grad-CAM 做特征选择 |
| 9 | Hui et al. (2026) | MSCAC-TCAM: An Interpretable Fault Diagnosis Method | IEEE Sensors J. | 新 | 提到了物理一致性分析但侧重于多传感器融合 |

## 关键论文详解

### 1. Chen & Lee (2020) — 开创性工作

- **方法**: 在 STFT 频谱图上训练 CNN，对每类故障生成 Grad-CAM 热力图
- **发现**: 不同故障类型的 Grad-CAM 高亮频率带不同，与理论故障频率部分吻合
- **未做**: 没有量化频带内显著性占比，没有负对照

### 2. Mey & Neufeld (2022) — 方法论框架

- **方法**: 提出了评估 XAI 方法在故障检测中的适用性的框架
- **关键贡献**: 强调 "需分析 Grad-CAM 高亮的频率带是否对应物理故障特征"
- **未做**: 提出了方法论但未自己执行负对照实验

### 3. Li et al. (2023) — 最相关

- **方法**: 提出 Multilayer Grad-CAM（MLG-CAM），从多层特征图生成热力图
- **量化指标**:
  - **RATM** (Relative Activation of Target Map): 目标区域显著性占比
  - **RATA** (Relative Activation of Target Area): 类似指标
  - **CEI** (Comprehensive Evaluation Indicator): 综合评价指标
- **关键发现**: "emphasizes fault characteristic frequency in frequency domain"
- **未做**: 聚焦于可视化改进而非物理一致性；未做频带平移负对照；未做跨种子/跨条件稳定性

### 4. Lu et al. (2023) — 健康库方法

- **方法**: 用 Grad-CAM 构建"健康库"（health library），在推理时检索相似特征图
- **发现**: "the proposed method can select prediction basis samples that are intuitively and physically meaningful"
- **未做**: 声称"physical meaningful"但未做定量物理一致性验证

## 研究缺口确认

**核心缺口**: 现有工作存在以下共同不足：

1. **仅定性展示，缺乏定量分析**：绝大多数论文只展示 Grad-CAM 热力图作为"证据"，没有计算故障频带内显著性占比
2. **没有负对照**：没有一篇论文设计了频带平移的负对照来排除"巧合"可能的解释
3. **没有稳定性分析**：没有分析跨种子、跨窗口、跨噪声条件下的 Grad-CAM 一致性
4. **没有将物理一致性与分类性能结合**：未构建"物理一致性 × 准确率"双维度评估矩阵

**我们的贡献点**:
- 首次在轴承故障诊断中引入 Grad-CAM 物理一致性的量化评估
- 首次设计频带平移负对照作为 GAD-CAM 解释有效性的检验
- 首次系统分析跨种子/跨窗口/跨噪声的 Grad-CAM 稳定性
