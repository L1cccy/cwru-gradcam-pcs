# 数据集清单 v2 — 方向五：Grad-CAM 物理一致性量化

## 双数据集概览

| 属性 | CWRU | MFPT |
|------|------|------|
| 原采样率 | 12 kHz (DE), 48 kHz (Normal) | 97.656 kHz / 48.828 kHz |
| 统一采样率 | 12 kHz (Normal 降采样 4×) | 12 kHz (全部降采样) |
| 类别 | Normal, InnerRace, OuterRace | Normal, InnerRace, OuterRace |
| 信号类型 | 加速度振动 | 加速度振动 |
| 传感器 | 驱动端加速度计 | 加速度计 |
| 故障植入方式 | 电火花加工 | 电火花加工 |

## CWRU 12kHz DE Data (3 classes)

### Normal (48kHz → 12kHz downsampled)

| 文件 | 转速 | Load | recording_id |
|------|------|------|-------------|
| N_0.mat | 1797 | 0 HP | Normal_N_0 |
| N_1.mat | 1772 | 1 HP | Normal_N_1 |
| N_2.mat | 1750 | 2 HP | Normal_N_2 |
| N_3.mat | 1730 | 3 HP | Normal_N_3 |

### InnerRace

| 故障尺寸 | 文件数 | Load范围 |
|----------|--------|---------|
| 0.007" | 4 | 0-3 HP |
| 0.014" | 4 | 0-3 HP |
| 0.021" | 4 | 0-3 HP |
| 0.028" | 4 | 0-3 HP |
| **合计** | **16** | |

### OuterRace (Centered)

| 故障尺寸 | 文件数 | Load范围 |
|----------|--------|---------|
| 0.007" | 4 | 0-3 HP |
| 0.014" | 4 | 0-3 HP |
| 0.021" | 4 | 0-3 HP |
| **合计** | **12** | |

## MFPT Data (3 classes)

### Normal (baseline)

| 文件 | 采样率 | Load | recording_id |
|------|--------|------|-------------|
| baseline_1.mat | 97.656 kHz | 270 lbs | MFPT_N_1 |
| baseline_2.mat | 97.656 kHz | 270 lbs | MFPT_N_2 |
| baseline_3.mat | 97.656 kHz | 270 lbs | MFPT_N_3 |

### InnerRace (vload)

| 文件 | 原采样率 | Load | recording_id |
|------|---------|------|-------------|
| vload_1 | 48.828 kHz | 0 lbs | MFPT_IR_1 |
| vload_2 | 48.828 kHz | 50 lbs | MFPT_IR_2 |
| vload_3 | 48.828 kHz | 100 lbs | MFPT_IR_3 |
| vload_4 | 48.828 kHz | 150 lbs | MFPT_IR_4 |
| vload_5 | 48.828 kHz | 200 lbs | MFPT_IR_5 |
| vload_6 | 48.828 kHz | 250 lbs | MFPT_IR_6 |
| vload_7 | 48.828 kHz | 300 lbs | MFPT_IR_7 |

### OuterRace

| 文件 | 采样率 | Load | recording_id |
|------|--------|------|-------------|
| Fault_1 | 97.656 kHz | 270 lbs | MFPT_OR_1 |
| Fault_2 | 97.656 kHz | 270 lbs | MFPT_OR_2 |
| Fault_3 | 97.656 kHz | 270 lbs | MFPT_OR_3 |
| vload_1 | 48.828 kHz | 25 lbs | MFPT_OR_4 |
| vload_2 | 48.828 kHz | 50 lbs | MFPT_OR_5 |
| vload_3 | 48.828 kHz | 100 lbs | MFPT_OR_6 |
| vload_4 | 48.828 kHz | 150 lbs | MFPT_OR_7 |
| vload_5 | 48.828 kHz | 200 lbs | MFPT_OR_8 |
| vload_6 | 48.828 kHz | 250 lbs | MFPT_OR_9 |
| vload_7 | 48.828 kHz | 300 lbs | MFPT_OR_10 |

## 段数估算

| 数据集 | Normal | InnerRace | OuterRace | 总段数 |
|--------|--------|-----------|-----------|--------|
| CWRU | ~944 | ~3776 | ~2832 | **~7552** |
| MFPT | ~2334 | ~5713 | ~6486 | **~14533** |
| **合计** | | | | **~22085** |

> 注：CWRU 排除 Ball 类后段数减少约 3776（Ball 被排除）；MFPT 段数 = 信号长度 / 512 × 窗口列系数

## 窗口切分

```
window_length = 1024 点
hop_length = 512 点 (50% overlap)
n_fft = 512 (频率分辨率 23.44 Hz)
```

## 跨数据集划分方案

- **CWRU**: 按 recording_id 分组 80/20
- **MFPT**: 按 recording_id 分组 80/20
- **跨数据集**: CWRU_train → MFPT_test（跨数据集泛化测试）
- **跨工况**: MFPT vload 按 load 分组 → 低负载 train / 高负载 test
