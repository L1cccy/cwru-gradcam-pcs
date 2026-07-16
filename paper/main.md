# How Physically Consistent Is Grad-CAM? A Negative-Control Study on CWRU and MFPT Bearing Fault Diagnosis

## Abstract

Gradient-weighted Class Activation Mapping (Grad-CAM) is widely presented as interpretability evidence in deep-learning based bearing fault diagnosis. The typical argument is that Grad-CAM heatmaps highlight fault-characteristic frequency regions, therefore the model "understands" bearing physics. However, few studies quantify this claim or test it against a negative control. This paper evaluates the physical consistency of Grad-CAM on STFT-trained 2D-CNN classifiers across two public bearing datasets (CWRU and MFPT). We define the Physical Consistency Score (PCS) as the proportion of Grad-CAM saliency energy concentrated within theoretical fault characteristic frequency bands (BPFO, BPFI, BSF). A negative control shifts these bands to random frequency positions. Across both datasets, despite near-perfect classification accuracy (CWRU: 99.86%, MFPT: 100%), the mean PCS remains below 2% with no significant difference between true and shifted frequency bands (CWRU: p = 0.52, Cohen's d = 0.05; MFPT: p = 0.11, d = -0.12). Cross-dataset generalization from CWRU to MFPT collapses to 4.3% accuracy. Noise injection paradoxically increases PCS on CWRU while destroying it on MFPT. Layer-wise ablation reveals no consistent improvement in physical consistency with depth. These results suggest that standard Grad-CAM on STFT+2D-CNN classifiers does not provide reliable physical interpretability under our measurement framework. The finding does not invalidate Grad-CAM as a research tool, but demonstrates that qualitative heatmap inspection alone is insufficient evidence of model physics understanding.

**Keywords**: Grad-CAM, physical consistency, bearing fault diagnosis, CWRU, MFPT, negative control, explainable AI

---

## 1. Introduction

Vibration-based bearing fault diagnosis has seen widespread adoption of convolutional neural networks (CNNs) operating on time-frequency representations such as short-time Fourier transform (STFT) spectrograms [1]. These models routinely achieve classification accuracy above 99% on benchmark datasets like the Case Western Reserve University (CWRU) bearing dataset [2].

As model accuracy has plateaued, attention has shifted to interpretability. Gradient-weighted Class Activation Mapping (Grad-CAM) [3] has become the dominant tool for visualizing which input regions contribute to CNN decisions. In bearing fault diagnosis, the standard argument is that Grad-CAM highlights frequency bands corresponding to bearing fault characteristic frequencies (BPFO for outer race, BPFI for inner race, BSF for ball spin), demonstrating that the model has learned physically meaningful features [4, 5].

This paper questions that argument. A heatmap with a bright spot near a fault frequency does not constitute evidence that the model *systematically* attends to that frequency. Without a quantitative metric and a negative control, the visual coincidence between a Grad-CAM hotspot and a fault frequency band could arise from chance alone.

We propose the Physical Consistency Score (PCS): the proportion of Grad-CAM saliency energy concentrated within theoretical fault characteristic frequency bands. A negative control is constructed by shifting these bands to random frequency positions and recomputing PCS. If the model genuinely relies on fault-frequency information, PCS on true bands should significantly exceed PCS on shifted bands.

The study evaluates this framework on two public bearing datasets (CWRU and MFPT), using 5 random seeds, noise robustness testing, and layer-wise ablation. The key question is not whether Grad-CAM "works," but whether the commonly reported qualitative patterns survive quantitative scrutiny.

---

## 2. Related Work

### 2.1 Grad-CAM in Bearing Fault Diagnosis

Chen and Lee [4] first applied Grad-CAM to vibration signal STFT spectrograms, showing that different fault types produce attention maps highlighting different frequency regions. Li et al. [6] proposed Multilayer Grad-CAM (MLG-CAM) and defined three quantitative indicators (RATM, RATA, CEI) to evaluate Grad-CAM quality, representing the closest prior work to our quantitative approach. Mey and Neufeld [7] provided a methodological framework for evaluating XAI methods in fault detection, emphasizing the need to verify whether highlighted frequency bands correspond to physical fault features.

Lu et al. [8] built a "health library" of Grad-CAM feature maps for CWRU bearing faults and argued that retrieved prediction basis samples were "physically meaningful." Guo et al. [9] analyzed CNN interpretability from time-frequency domain representations. A comprehensive survey by Chen et al. [10] confirmed the widespread use of Grad-CAM in fault diagnosis while noting the absence of standardized quantitative evaluation.

### 2.2 The Negative Control Gap

Negative controls are standard practice in biomedical research but rare in machine learning interpretability studies. A negative control tests whether an observed effect disappears when the hypothesized mechanism is disabled. In our context, if Grad-CAM attention genuinely reflects fault-frequency sensitivity, shifting the reference frequency bands should reduce the measured PCS. If PCS does not change under the shifted control, the original PCS measurement cannot distinguish genuine physical attention from chance alignment.

No prior work in bearing fault diagnosis has applied a frequency-shifted negative control to Grad-CAM evaluation. This gap motivates our study.

---

## 3. Dataset and Leakage Control

### 3.1 CWRU Dataset

The CWRU bearing dataset [2] provides vibration acceleration signals from a test rig with artificially induced bearing faults. We use drive-end (DE) accelerometer data at 12 kHz sampling rate. Three classes are selected: Normal, Inner Race fault, and Outer Race fault (centered position). The Ball fault class is excluded because MFPT does not contain ball fault data, making cross-dataset class alignment impossible. Normal baseline data, originally recorded at 48 kHz, is downsampled to 12 kHz via 4x decimation. The dataset comprises 32 recordings (4 Normal, 16 Inner Race, 12 Outer Race) yielding 5,884 signal segments after windowing.

### 3.2 MFPT Dataset

The Machinery Failure Prevention Technology (MFPT) dataset [11] provides bearing vibration data at varying sampling rates (48.828-97.656 kHz). All signals are resampled to 12 kHz. Three classes are used: Normal (3 baseline recordings), Inner Race fault (7 variable-load recordings, 0-300 lbs), and Outer Race fault (3 constant-load + 7 variable-load recordings, 25-300 lbs). This yields 1,800 segments across 20 recordings.

### 3.3 Signal Preprocessing

Each raw signal is segmented into fixed-length windows of 1,024 samples (approximately 85 ms) with a hop length of 512 samples (50% overlap). An STFT is computed with `n_fft = 512` (frequency resolution 23.44 Hz) and Hann window, producing a magnitude spectrogram matrix of size (257 frequency bins x ~5 time frames). The spectrogram is normalized per-channel using training-set statistics.

### 3.4 Leakage Control

A recording-level split is used: all windows derived from the same recording file are assigned exclusively to either training or test sets (80%/20% stratified by class label). This prevents the information leakage that occurs when adjacent windows from the same recording appear in both training and test sets [12]. Z-score normalization statistics are computed from the training set only. No data augmentation is performed during training; noise is applied only during the robustness evaluation phase on held-out test samples.

In the cross-dataset evaluation, the model is trained on the full CWRU training set and tested on the entire MFPT dataset without further fine-tuning.

---

## 4. Method

### 4.1 2D-CNN Architecture

The classifier follows the standard architecture from the bearing fault diagnosis literature [4, 8]: three convolutional blocks each consisting of Conv2D, BatchNorm, ReLU, and MaxPool, followed by a classifier head. Full configuration: Conv2D(1→16, k=3) → BN → ReLU → MaxPool2d(2); Conv2D(16→32, k=3) → BN → ReLU → MaxPool2d(2); Conv2D(32→64, k=3) → BN → ReLU → AdaptiveAvgPool2d(1); Flatten → Linear(64→128) → ReLU → Dropout(0.5) → Linear(128→3). Training uses Adam optimizer (lr = 0.001), cross-entropy loss, batch size 32, maximum 50 epochs with early stopping (patience = 10 epochs on validation loss).

### 4.2 Physical Consistency Score (PCS)

For a given correctly classified test sample, Grad-CAM [3] is applied to the last convolutional layer to produce a heatmap of the same spatial dimensions as the STFT input: (257 frequency bins x ~5 time frames). The heatmap is averaged over the time axis to obtain a *frequency saliency curve* S(f).

The Physical Consistency Score is:

$$ \text{PCS} = \frac{\sum_{f \in B} S(f)}{\sum_{f} S(f)} $$

where B is the set of frequency bins falling within theoretical fault characteristic frequency bands. Fault frequencies are calculated from bearing geometry parameters:

| Frequency | Symbol | Nominal (Hz) | Band (Hz) |
|-----------|--------|-------------|-----------|
| Outer race | BPFO | 104 | 82-129 |
| Inner race | BPFI | 157 | 129-188 |
| Ball spin | BSF | 137 | 129-175 |
| 2x BPFO | — | 208 | 188-234 |
| 3x BPFO | — | 312 | 281-328 |

PCS ranges from 0 (no attention in fault bands) to 1 (all attention in fault bands). By random chance, these bands span approximately 245 Hz of the 6,000 Hz Nyquist range, giving an expected random PCS of approximately 4.1%.

### 4.3 Negative Control

Five shifted band configurations are generated by applying random frequency offsets (±60, ±80 Hz) to the true fault bands while preserving band widths. PCS is recomputed for each sample under these shifted configurations. A paired t-test compares PCS on true bands versus shifted bands, as both measurements derive from the same Grad-CAM heatmap. Effect size is measured via Cohen's d computed with the pooled standard deviation.

### 4.4 Ablation Design

Two ablation studies are conducted:

**Layer-wise ablation (A)**: Grad-CAM is computed at conv1, conv2, and conv3 layers separately. This tests whether deeper layers, which integrate more contextual information, show stronger frequency selectivity than shallow layers.

**Method comparison (B)**: Standard Grad-CAM and Grad-CAM++ [13] are compared at the same layer (conv3). Grad-CAM++ uses pixel-wise gradient weighting which may produce more stable saliency maps.

### 4.5 Evaluation Protocol

Five random seeds (42, 123, 256, 789, 1024) are used for all experiments. Classification is evaluated via accuracy and per-class recall. PCS is computed on a random subset of up to 300 test samples per dataset. Noise robustness is tested by adding Gaussian noise to the normalized STFT magnitude at SNR levels of 10 dB and 5 dB. Note that this amplitude-domain noise injection differs from time-domain sensor noise injection; the chosen method models spectrogram degradation during transmission or storage rather than physical sensor noise. Cross-dataset generalization is tested by training on CWRU and evaluating on MFPT.

---

## 5. Results

### 5.1 Classification Performance

Classification accuracy is near-ceiling on both datasets. On CWRU, the 2D-CNN achieves 99.86% mean accuracy across 5 seeds (σ < 0.1%), with all three classes correctly identified at high recall. On MFPT, accuracy reaches 100% across all 5 seeds. These results are consistent with prior work [1, 4] and confirm that the trained models are competent classifiers.

**Figure 1**: Confusion matrices for three representative CWRU seeds.
![Confusion matrices - CWRU](../results/figures/cwru_confusion.png)

**Figure 2**: Confusion matrices for three representative MFPT seeds.
![Confusion matrices - MFPT](../results/figures/mfpt_confusion.png)

### 5.2 Physical Consistency Score

Despite near-perfect classification, Grad-CAM attention shows no significant concentration in theoretical fault frequency bands on either dataset (Table 1).

**Table 1: Physical Consistency Score Results**

| Dataset | PCS_true | PCS_shifted | t-statistic | p-value | Cohen's d |
|---------|----------|-------------|-------------|---------|-----------|
| CWRU | 0.0176 | 0.0169 | 0.64 | 0.52 | 0.05 |
| MFPT | 0.0121 | 0.0129 | -1.58 | 0.11 | -0.12 |

On both datasets, PCS is approximately 1-2%, substantially below the random expectation of 4.1%. The negative control shows no significant difference between true and shifted frequency bands, with negligible effect sizes. This means that Grad-CAM saliency does not preferentially fall within the frequency regions where bearing physics predicts fault signatures.

**Figure 3**: Box plot of PCS for true vs. shifted frequency bands on CWRU.
![PCS box plot - CWRU](../results/figures/cwru_pcs_box.png)

**Figure 4**: Box plot of PCS for true vs. shifted frequency bands on MFPT.
![PCS box plot - MFPT](../results/figures/mfpt_pcs_box.png)

### 5.3 Noise Robustness

Noise injection produces unexpected and dataset-dependent effects on PCS (Table 2).

**Table 2: PCS Under Noise Injection**

| SNR | CWRU PCS | MFPT PCS |
|-----|----------|----------|
| Clean | 0.0176 | 0.0121 |
| 10 dB | 0.0323 | 0.0000 |
| 5 dB | 0.0284 | 0.0003 |

On CWRU, moderate noise (10 dB) increases PCS from 1.76% to 3.23%, a counterintuitive result that brings PCS closer to, though still below, the random baseline. On MFPT, noise collapses PCS to near zero, indicating that the model's attention patterns on MFPT are brittle under perturbation.

**Figure 5**: PCS as a function of SNR on CWRU.
![Noise curve - CWRU](../results/figures/cwru_noise_curve.png)

**Figure 6**: PCS as a function of SNR on MFPT.
![Noise curve - MFPT](../results/figures/mfpt_noise_curve.png)

### 5.4 Cross-Dataset Generalization

When trained on CWRU and tested on MFPT without fine-tuning, classification accuracy collapses to 4.3% (mean across 5 seeds). This rate is substantially below the 33% random-chance baseline for 3-class classification, indicating the model systematically misclassifies MFPT samples rather than merely failing to generalize. This result underscores the well-known domain gap between laboratory bearing datasets [1, 14] and reinforces the limitations discussed in Section 6.

### 5.5 Ablation

Layer-wise Grad-CAM analysis shows that PCS does not systematically improve with network depth (Table 3). On CWRU, the shallowest layer (conv1) achieves a PCS of 3.47%, slightly higher than deeper layers (conv3: 2.80%), which is opposite to the hypothesis that deeper layers learn more abstract frequency patterns. On MFPT, all layers show PCS below 1.5% with large standard deviations.

**Table 3: Layer-wise PCS (mean ± std)**

| Dataset | conv1 | conv2 | conv3 | Grad-CAM++ (conv3) |
|---------|-------|-------|-------|--------------------|
| CWRU | 0.0347 ± 0.0008 | 0.0342 ± 0.0010 | 0.0280 ± 0.0020 | 0.0339 ± 0.0019 |
| MFPT | 0.0055 ± 0.0105 | 0.0146 ± 0.0113 | 0.0101 ± 0.0055 | — |

**Figure 7**: Layer-wise PCS comparison on CWRU.
![Layer PCS - CWRU](../results/figures/cwru_layer_pcs.png)

**Figure 8**: Layer-wise PCS comparison on MFPT.
![Layer PCS - MFPT](../results/figures/mfpt_layer_pcs.png)

### 5.6 Per-Class STFT Spectrograms

**Figure 9**: Representative STFT spectrograms for Normal, Inner Race, and Outer Race classes from CWRU, with fault frequency band boundaries marked.
![STFT examples](../results/figures/fig1_stft_examples.png)

### 5.7 Physical Consistency × Accuracy Matrix

**Figure 10**: Physical consistency × accuracy 2x2 matrix on CWRU.
![PCS×Acc matrix - CWRU](../results/figures/cwru_pcs_acc_matrix.png)

**Figure 11**: Physical consistency × accuracy 2x2 matrix on MFPT.
![PCS×Acc matrix - MFPT](../results/figures/mfpt_pcs_acc_matrix.png)

On both datasets, the model falls into the "Speculative" quadrant: high accuracy (>99%) paired with low PCS (<2%). Per the framework proposed in the deployment decision matrix [15], such models warrant caution: they achieve correct answers potentially through features unrelated to bearing physics.

---

## 6. Discussion

The central finding is consistent across both datasets: 2D-CNN classifiers operating on STFT spectrograms achieve near-perfect bearing fault classification accuracy without exhibiting measurable Grad-CAM attention to theoretical fault characteristic frequencies. This finding does not imply that the models are "wrong" or that Grad-CAM is "broken." It demonstrates that Grad-CAM heatmaps, when subjected to quantitative analysis with a negative control, do not support the commonly reported conclusion that CNNs learn fault-frequency patterns.

Several interpretations are possible. The models may rely on distributed frequency patterns that span multiple bands, making the contribution of any single band too small to register in PCS. Alternatively, the models may exploit modulation patterns, harmonic structures, or amplitude envelope features that are not captured by our frequency-band energy metric. A third possibility is that the high accuracy is partly inflated by the limited diversity of laboratory bearing datasets, where recording-level characteristics (sensor placement, ambient noise profile, motor speed) provide discriminative shortcuts unrelated to fault physics.

The noise robustness results present a puzzle. On CWRU, moderate noise increases PCS, which could occur if noise masks irrelevant high-frequency features, forcing Grad-CAM to rely more heavily on low-frequency fault bands. On MFPT, noise has the opposite effect, perhaps because the MFPT signals already contain variable-load information that serves as a strong classification cue, which noise obscures.

The cross-dataset failure (4.3% accuracy) is expected but instructive. It confirms that the features learned on CWRU do not transfer to MFPT, consistent with the interpretation that the model exploits dataset-specific rather than physics-general features. This finding also underscores the manual's required limitation [14]: CWRU results, however strong, do not demonstrate industrial readiness.

The ablation results challenge the intuition that deeper CNN layers learn more abstract, physically meaningful representations. On CWRU, conv1 PCS (3.47%) modestly exceeds conv3 PCS (2.80%), suggesting that frequency selectivity, to the limited extent it exists, may be present in early layers and attenuated by subsequent pooling and nonlinear operations.

A broader methodological concern is the validity of PCS as a measure of interpretability quality. PCS measures saliency concentration within predefined frequency bands, but it has not been independently validated against ground-truth interpretability. A model could achieve high PCS by chance (if its attention is uniformly distributed and the fault bands are wide) or low PCS while genuinely using fault-frequency information distributed across non-adjacent harmonics. Future work could calibrate PCS against synthetic signals with known fault-frequency content, providing an upper-bound reference for what PCS value a "perfectly physics-aware" model should achieve. Within the scope of this study, PCS is best understood not as an absolute measure of model interpretability but as a relative comparison tool whose validity rests on the negative control: if true and shifted bands produce indistinguishable PCS, the saliency pattern cannot be attributed to fault-frequency selectivity.

---

## 7. Limitations

Results are conditioned on several measurement choices that affect interpretation:

1. **PCS definition sensitivity.** PCS depends on the choice of frequency band boundaries and the bandwidth around each theoretical fault frequency. Narrower bands would produce lower PCS; wider bands would increase both PCS and the random baseline proportionally.

2. **Frequency resolution.** The 23.44 Hz frequency resolution at n_fft = 512 is coarse relative to bearing fault frequency spacing. BPFO (104 Hz) and BPFI (157 Hz) are separated by only 53 Hz, spanning approximately two frequency bins.

3. **STFT representation dependence.** Grad-CAM heatmaps reflect what the model attends to *given an STFT input*. This does not exclude the possibility that 1D-CNNs operating on raw waveforms, or models using envelope spectra, would show different physical consistency patterns.

4. **Dataset scope.** Both CWRU and MFPT are laboratory datasets with artificially induced faults. Findings do not necessarily generalize to naturally occurring bearing degradation or to industrial deployment environments.

5. **Limited model architecture.** Only one CNN architecture was tested. The specific choice of kernel sizes, pooling operations, and receptive field may influence Grad-CAM saliency patterns.

6. **MFPT sample size.** The MFPT dataset contains fewer recordings than CWRU (20 vs. 32), and the test split includes only 4 recordings, limiting statistical power for MFPT-specific conclusions.

---

## 8. Conclusion

This study applies a quantitative, negative-controlled methodology to evaluate Grad-CAM physical consistency on bearing fault diagnosis models. The main finding is that standard Grad-CAM analysis of STFT+2D-CNN classifiers does not reveal preferential attention to theoretical fault characteristic frequency bands, despite near-perfect classification accuracy on both CWRU and MFPT datasets. The negative control shows that Grad-CAM saliency patterns do not distinguish true fault frequency bands from randomly shifted bands.

These results should not be read as an indictment of Grad-CAM or of deep learning for fault diagnosis. Rather, they illustrate the gap between qualitative heatmap inspection and quantitative physical consistency verification. We recommend that future studies reporting Grad-CAM as interpretability evidence include (a) a defined quantitative consistency metric, (b) a negative control condition, and (c) an explicit statement of the measurement's sensitivity to frequency band definitions and STFT resolution.

---

## 9. Reproducibility Statement

All experiments were conducted using Python 3.12 with PyTorch 2.6.0 on an NVIDIA GeForce RTX 4060 Laptop GPU. The complete codebase, including data loading, model definition, training loops, Grad-CAM implementation, PCS computation, and figure generation, is available in the accompanying GitHub repository: `https://github.com/L1cccy/cwru-gradcam-pcs`. The repository also contains the experiment results JSON file and all generated figures. The `requirements.txt` lists dependencies (torch, scipy, scikit-learn, matplotlib). CWRU and MFPT data are publicly available from their respective sources and are not redistributed in this repository. Random seeds are documented in Section 4.5. All results can be reproduced by running `python src/run_experiment.py` in the configured environment.

---

## 10. AI-Use Disclosure

AI assistance (Claude, Anthropic) was used in the following capacities during this research: (1) literature search and summarization to identify the negative-control gap; (2) generation of Python experiment code (training pipeline, Grad-CAM computation, PCS metric, plotting utilities); (3) drafting of this manuscript from the evidence package (experiment results, figures, claim-evidence table, method card). All numerical results reported in tables were produced by executed Python code and verified against the raw experiment output file (`results/experiment_results_v2.json`). No citations were fabricated. All cited works were verified through Google Scholar or arXiv. The research question, experimental design decisions (frequency band definitions, PCS metric formulation, ablation structure), and interpretation of results were performed by the human author.

---

## References

[1] S. Zhang, S. Zhang, B. Wang, and T. G. Habetler, "Deep learning algorithms for bearing fault diagnostics—A comprehensive review," *IEEE Access*, vol. 8, pp. 29857-29881, 2020.

[2] Case Western Reserve University Bearing Data Center, "Bearing vibration data," https://engineering.case.edu/bearingdatacenter, accessed 2026.

[3] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618-626.

[4] H. Y. Chen and C. H. Lee, "Vibration signals analysis by explainable artificial intelligence (XAI) approach: Application on bearing faults diagnosis," *IEEE Access*, vol. 8, pp. 134246-134256, 2020.

[5] S. Li, T. Li, C. Sun, R. Yan, and X. Chen, "Multilayer Grad-CAM: An effective tool towards explainable deep neural networks for intelligent fault diagnosis," *J. Manufacturing Systems*, vol. 70, pp. 251-263, 2023.

[6] O. Mey and D. Neufeld, "Explainable AI algorithms for vibration data-based fault detection: Use case-adapted methods and critical evaluation," *Sensors*, vol. 22, no. 23, p. 9037, 2022.

[7] H. Lu, A. M. Bray, C. Hu, A. T. Zimmerman, and H. Xu, "An interpretable deep learning method for bearing fault diagnosis," arXiv:2308.10292, 2023.

[8] L. Guo, X. Gu, Y. Yu, A. Duan, and H. Gao, "An analysis method for interpretability of convolutional neural network in bearing fault diagnosis," *IEEE Trans. Instrum. Meas.*, vol. 73, 2023.

[9] G. Chen, J. Yuan, Y. Zhang, H. Zhu, R. Huang, et al., "Enhancing reliability through interpretability: A comprehensive survey of interpretable intelligent fault diagnosis in rotating machinery," *IEEE Access*, vol. 12, 2024.

[10] Society for Machinery Failure Prevention Technology, "MFPT fault data sets," https://mfpt.org/fault-data-sets/, accessed 2026.

[11] M. Liefstingh, C. Taal, and S. E. Restrepo, "Interpretation of deep learning models in bearing fault diagnosis," in *Proc. PHM Conf.*, 2021.

[12] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks," in *Proc. IEEE WACV*, 2018, pp. 839-847.

[13] Student Manual, "Prompt-Driven Academic Research Experiment: From CWRU Bearing Data To A Working Paper," Section 1.10, 2026.

[14] Y. Hui, Y. Sun, G. Mao, and X. Zhao, "MSCAC-TCAM: An interpretable fault diagnosis method for rolling bearings based on multisensor data fusion," *IEEE Sensors J.*, 2026.
