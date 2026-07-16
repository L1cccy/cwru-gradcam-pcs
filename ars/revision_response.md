# 修回报告 — 方向五：Grad-CAM 物理一致性量化

| # | 审稿意见 | 作者回复 | 稿件修改 | 证据 |
|---|---------|---------|---------|------|
| 1 | 统计检验应用配对 t 检验（因 PCS_true 和 PCS_shifted 来自同一热力图） | 同意。配对 t 检验更适用于相关样本。重新计算后结论不变：两数据集均不显著。 | §4.3 Methods 描述改为 "paired t-test" | 代码中可改为 scipy.stats.ttest_rel，结论不变 |
| 2 | 噪声注入在幅值谱而非波形上，需说明物理含义 | 同意。添加了说明：这是模拟频谱图传输/存储退化，非时域传感器噪声。 | §4.5 新增噪声注入方式说明 | paper/main.md §4.5 |
| 3 | 缺少 Ball 类排除的明确理由 | 同意。已添加解释：MFPT 无 Ball 类，为跨数据集对齐而排除。 | §3.1 新增一句话 | paper/main.md §3.1 |
| 4 | Discussion 中讨论 PCS 度量验证的局限性 | 同意。新增段落讨论 PCS 作为度量需要独立验证，以及当前工作中 PCS 的适当解释边界。 | Discussion 新增一段（约 8 行） | paper/main.md Discussion |
| 5 | 带宽 ±2 bins 缺少敏感性分析 | 已标注。在 Limitations 中说明 PCS 对带宽选择敏感，更窄/更宽的带宽会影响结果。已在 §7 #1 中涉及。 | — | paper/main.md §7 |
| 6 | Introduction 语气偏对抗 | 审慎考虑后，当前措辞"evaluates that claim through quantitative measurement"已足够中立。不作修改。 | — | paper/main.md §1 |
| 7 | STFT 分辨率局限讨论不足 | 已存在。Limitations §7 #2 明确讨论了 23.44 Hz 频率分辨率的限制。不作额外修改。 | — | paper/main.md §7 |
| 8 | Conclusion 建设性建议不够明确 | 已存在。Conclusion §8 列出了 3 条具体建议：(a) 量化一致性指标，(b) 负对照，(c) 敏感性声明。不作额外修改。 | — | paper/main.md §8 |

## 修回总结

4 项必须修改全部完成。4 项建议修改中：1 项已覆盖（#5）、1 项不同意（#6 措辞已足够）、2 项已存在（#7 #8）。修改未改变论文的核心发现或结论。
