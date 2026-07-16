# 终审诚信检查报告 — 方向五：Grad-CAM 物理一致性量化

## 逐项核实

### 1. 论文中的每条数字结果是否都能在 results/ 的表格或日志中找到？

| 论文声明 | 来源 | 核实 |
|---------|------|------|
| CWRU准确率 99.86% (σ<0.1%) | results/experiment_results_v2.json cwru.accuracies | ✅ 5种子: [0.9985, 0.9985, 0.9985, 0.9985, 0.9992] |
| MFPT准确率 100% | results/experiment_results_v2.json mfpt.accuracies | ✅ 5种子: [1.0, 1.0, 1.0, 1.0, 1.0] |
| CWRU PCS_true=0.0176, PCS_shifted=0.0169 | results/experiment_results_v2.json cwru.pcs_true, cwru.pcs_shifted | ✅ |
| CWRU t=0.64, p=0.52, d=0.05 | results/experiment_results_v2.json cwru | ✅ |
| MFPT PCS_true=0.0121, PCS_shifted=0.0129 | results/experiment_results_v2.json mfpt | ✅ |
| MFPT t=-1.58, p=0.11, d=-0.12 | results/experiment_results_v2.json mfpt | ✅ |
| CWRU→MFPT 4.3% | results/experiment_results_v2.json cross_dataset | ✅ |
| CWRU噪声 PCS: 10dB=0.0323, 5dB=0.0284 | results/experiment_results_v2.json cwru.noise_pcs | ✅ |
| MFPT噪声 PCS: 10dB=0.00, 5dB=0.0003 | results/experiment_results_v2.json mfpt.noise_pcs | ✅ |
| 层消融值 | results/experiment_results_v2.json cwru.layer_pcs, mfpt.layer_pcs | ✅ |

**结论**: ✅ PASS — 全部数字可追踪

### 2. 每条引用是否真实存在且支撑对应论述？

| 引用 | DOI/URL | 内容核实 | 状态 |
|------|---------|---------|------|
| [1] Zhang et al. 2020 | IEEE Access | Survey paper, 领域综述 | ✅ VERIFIED |
| [2] CWRU Bearing Data | engineering.case.edu | 公开数据集 | ✅ VERIFIED |
| [3] Selvaraju et al. 2017 (Grad-CAM) | ICCV 2017 | 原始 Grad-CAM 论文 | ✅ VERIFIED |
| [4] Chen & Lee 2020 | IEEE Access | XAI+CWRU | ✅ VERIFIED |
| [5] Li et al. 2023 (MLG-CAM) | JMS | 多层 Grad-CAM | ✅ VERIFIED |
| [6] Mey & Neufeld 2022 | Sensors | XAI评估框架 | ✅ VERIFIED |
| [7] Lu et al. 2023 | arXiv:2308.10292 | 健康库方法 | ✅ VERIFIED |
| [8] Guo et al. 2023 | IEEE TIM | CNN可解释性 | ✅ VERIFIED |
| [9] Chen et al. 2024 | IEEE Access | 综述 | ✅ VERIFIED |
| [10] MFPT | mfpt.org | 公开数据集 | ✅ VERIFIED |
| [11] Liefstingh et al. 2021 | PHM Conf. | Grad-CAM+轴承 | ✅ VERIFIED |
| [12] Chattopadhyay et al. 2018 (Grad-CAM++) | WACV 2018 | Grad-CAM++ | ✅ VERIFIED |
| [13] Student Manual | 课程材料 | §1.10 局限 | ⚠️ UNVERIFIED (内部材料) |
| [14] Hui et al. 2026 | IEEE Sensors J. | MSCAC-TCAM | ✅ VERIFIED |

**结论**: ✅ PASS — 13/14 核实, 1项为内部课程材料

### 3. 所有关于 CWRU 数据集的声明是否与数据清单一致？

| 声明 | 数据清单 | 核实 |
|------|---------|------|
| 12 kHz DE 采样率 | dataset_cards.md §CWRU | ✅ |
| Normal 48kHz→12kHz降采样 | dataset_cards.md | ✅ |
| 32 recordings | dataset_cards.md | ✅ (4N+16IR+12OR) |
| 3 类（Normal/IR/OR） | dataset_cards.md | ✅ |

**结论**: ✅ PASS

### 4. 是否存在 AI 生成的声明未经人工验证？

全部数字从实验结果 JSON 中提取，无编造。引文通过 Google Scholar 核实。AI 起草文本由作者审阅。

**结论**: ✅ PASS

### 5. Limitations 是否明确列出 CWRU 的固有局限？

论文 §7 列出 6 条局限，包含：
- 人工植入故障（#4）
- 实验室环境局限（#4）
- 单一模型架构（#5）

**结论**: ✅ PASS

### 6. 划分方案是否在论文中完整描述？

§3.4 描述了 recording-level split, 80/20, 分层采样, 训练集归一化, 无训练时增强。

**结论**: ✅ PASS

### 7. 负面结果或失败实验是否如实报告？

- 跨数据集失败（4.3%）✅ 如实报告
- PCS 与负对照无显著差异 ✅ 如实报告
- 噪声注入的异常行为 ✅ 如实报告

**结论**: ✅ PASS

### 8. AI 使用声明是否完整？

§10 AI-Use Disclosure 包含：使用了哪个 AI（Claude, Anthropic）、用于什么（文献/代码/草稿）、验证方式（数字从 JSON 核实）、未编造引用。

**结论**: ✅ PASS

---

## 7-Mode AI Research Failure Mode Checklist

| 模式 | 描述 | 状态 |
|------|------|------|
| M1 实现错误自查通过 | 代码运行成功，输出验证 | ✅ CLEARED |
| M2 幻觉引用 | 13/14 已核实 | ✅ CLEARED |
| M3 幻觉实验数据 | 所有数字 traceable to JSON | ✅ CLEARED |
| M4 依赖捷径 | 这恰恰是我们检验的问题 | ✅ CLEARED |
| M5 bug 被包装为发现 | 无证据 | ✅ CLEARED |
| M6 方法论伪造 | recording-level split 实际上确实实现了 | ✅ CLEARED |
| M7 早期框架锁定 | 实验前有完整实验协议 | ✅ CLEARED |

---

## 总体判定: ✅ PASS

所有 8 项检查通过，7 个失败模式均清除。准予进入最终发布阶段。
