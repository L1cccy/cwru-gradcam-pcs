# Academic Peer Review Report

## Structured Pruning Robustness Across Neural Architectures for Bearing Fault Diagnosis

---

# Phase 0 — Field Analysis & Reviewer Configuration

## Field Analysis

| Attribute | Determination |
|-----------|--------------|
| **Primary Discipline** | Mechanical Engineering / Condition Monitoring & Fault Diagnosis |
| **Secondary Discipline** | Machine Learning / Model Compression & Pruning |
| **Research Paradigm** | Empirical quantitative benchmarking with controlled experiments |
| **Methodology Type** | Comparative performance evaluation with ablation studies |
| **Target Journal Tier** | Mid-tier (e.g., *Mechanical Systems and Signal Processing*, *IEEE Transactions on Industrial Informatics*, *Engineering Applications of Artificial Intelligence*) |
| **Paper Maturity** | Working paper / technical report — solid experimental foundation, several gaps remain |
| **Manuscript Type** | Comparative study / benchmark report |

## Reviewer Configuration Card

| Reviewer | Role | Assigned Identity | Focus |
|----------|------|-------------------|-------|
| **EIC** | Editor-in-Chief | Editor from *Mechanical Systems and Signal Processing* — expertise in signal processing, fault diagnosis, and statistical methodology in engineering research | Journal fit, overall contribution, originality, significance to the fault diagnosis community |
| **R1 (Methodology)** | Peer Reviewer 1 | Researcher specializing in ML benchmarking methodology, experimental design for model compression, reproducibility, and statistical inference in engineering applications | Research design rigor, statistical validity of comparisons, reproducibility, ablation methodology |
| **R2 (Domain)** | Peer Reviewer 2 | Bearing fault diagnosis domain expert with deep knowledge of CWRU/MFPT benchmarks, vibration feature extraction, and SOTA diagnostic architectures | Literature coverage, theoretical grounding in fault diagnosis, practical relevance of findings, dataset choices |
| **R3 (Perspective)** | Peer Reviewer 3 | Cross-disciplinary researcher from the model compression & efficient ML systems community — works on pruning theory, sparse neural networks, and edge deployment | Pruning methodology choices, connection to broader pruning literature, deployment implications |
| **DA (Devil's Advocate)** | Devil's Advocate | Critical analyst charged with stress-testing the paper's core arguments, detecting logical fallacies, and identifying strongest counter-arguments | Core claim validation, alternative interpretations, overgeneralization risks |

---

# Phase 1 — Multi-Perspective Review Reports

---

## Report 1: Editor-in-Chief (EIC) Review

# Peer Review Report

## Manuscript Information
- **Title**: Structured Pruning Robustness Across Neural Architectures for Bearing Fault Diagnosis
- **Review Date**: July 14, 2026
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Editor-in-Chief

### Reviewer Identity
Editor from *Mechanical Systems and Signal Processing* — expertise in signal processing, condition monitoring, fault diagnosis, and statistical methodology. Familiar with both the CWRU benchmarking tradition and the growing interest in edge-deployable diagnostic models.

### Review Focus
Assessing journal fit, overall contribution, originality, and significance to the fault diagnosis community. Evaluating whether the paper addresses an important gap, whether the findings are credible, and whether the manuscript meets the journal's standards for publication.

---

## Overall Assessment

### Recommendation
- [ ] **Accept**
- [ ] **Minor Revision**
- [ ] **Major Revision**
- [x] **Reject** (Resubmit Encouraged — see rationale)

### Confidence Score
5 — Completely within my area of expertise, I am very confident in my assessment.

### Summary Assessment
This paper presents the first systematic comparison of structured pruning robustness across FC, 1D-CNN, and 2D-CNN architectures for bearing fault diagnosis on CWRU and MFPT datasets. The study addresses a genuine gap — practitioners choosing architectures for edge deployment currently have no cross-architecture pruning robustness guidance. The experimental design includes file-level splitting (avoiding a known leakage pitfall), multiple seeds, and an informative ablation comparing prune-then-fine-tune against training from scratch at reduced width. The finding that L1 zero-masking pruning is consistently inferior to training a smaller model from scratch is practically relevant, and the honest reporting of a falsified hypothesis (2D-CNN is NOT more pruning-robust) is commendable.

However, the paper in its current form has several issues that prevent me from recommending publication. The experimental scope is narrow — only one pruning method (L1 structured), only two datasets (both bearing fault, both laboratory benchmarks), and limited statistical power (3 seeds, no confidence intervals on the reported accuracies). The FC architecture at ~691K parameters is not a meaningful comparator to the CNN architectures (~24-32K). Most critically, the paper's central claim — that "prune-then-fine-tune is consistently inferior" — rests on an ablation where fine-tuning runs for only 10 epochs while the scratch model trains for 50, making the comparison potentially confounded by training budget rather than pruning damage. The hypothesis testing is informal (no statistical test), and the noise robustness and latency analyses are too superficial to support the paper's conclusions. Despite these concerns, the core idea is sound and the gap is real. I recommend **Major Revision with re-review**, or alternatively a targeted resubmission after substantial strengthening.

---

## Strengths

### S1: First systematic cross-architecture pruning comparison in bearing diagnosis
The paper correctly identifies a genuine gap: "no published study systematically compares how different neural architectures respond to identical pruning procedures on the same bearing fault diagnosis benchmark" (Section 1, p.1). This fills a concrete need for practitioners.

### S2: File-level splitting for leakage control
The use of file-level splitting (Section 3.3, p.3) rather than random window splitting is methodologically sound and addresses a known confound in bearing diagnosis benchmarking (Xu et al., 2022). This decision strengthens the validity of all reported results.

### S3: Honest reporting of a falsified hypothesis
The paper's Hypothesis (Section 1, p.2) — that 2D-CNN would be more pruning-robust — is clearly stated and then honestly reported as unsupported (Section 5.1, p.5). This intellectual honesty is a strength.

### S4: Informative ablation study
The comparison between prune-then-fine-tune and training from scratch at matched reduced width (Section 5.4) is the most novel contribution. The finding of gaps up to 32 percentage points is striking.

### S5: Reproducibility and AI-use disclosure
The reproducibility statement (Section 8) is detailed and actionable. The AI-use disclosure (Section 9) is transparent and appropriately scoped.

---

## Weaknesses

### W1: Confounded ablation — fine-tuning budget vs. pruning damage
**Problem**: The ablation (Section 5.4) compares prune-then-fine-tune (10 epochs fine-tuning at LR 0.0005) against training from scratch (50 epochs at LR 0.001 with scheduling). The scratch model receives 5× the training budget and a different learning rate schedule. The gap (up to 32 pp) may partially reflect insufficient fine-tuning rather than inherent pruning damage.
**Why it matters**: This confound undermines the paper's most striking claim. It changes the interpretation from "pruning damages model structure irreparably" to "pruning followed by insufficient fine-tuning performs worse than full training."
**Suggestion**: Either (a) match the total training budget (fine-tune for 50 epochs, or train the scratch model for only 10), or (b) explicitly acknowledge this confound and add experiments controlled for total gradient steps.
**Severity**: Critical

### W2: Insufficient statistical characterization
**Problem**: All results report only the mean across 3 seeds (Tables 1 and 2) with no standard deviations, confidence intervals, or per-seed breakdowns. Three seeds provide very limited statistical power. The hypothesis that 2D-CNN is not more pruning-robust is asserted without any statistical test (e.g., comparing the distribution of accuracy drops across seeds).
**Why it matters**: Readers cannot assess the reliability of reported performance differences. The 20.2 pp vs. 27.2 pp difference between 1D-CNN and 2D-CNN at 75% pruning on CWRU may or may not be significant with only 3 observations.
**Suggestion**: Report standard deviations or per-seed results. Add a simple statistical comparison (e.g., paired t-test or bootstrap confidence intervals on the accuracy drop). Increase to at least 5-10 seeds if feasible.
**Severity**: Critical

### W3: Narrow experimental scope limits generalizability
**Problem**: The paper uses only one pruning method (L1-norm structured channel pruning), two datasets (both bearing fault, both clean laboratory benchmarks), and one set of architectures (hand-designed, not including modern architectures like ResNet variants, transformers, or lightweight networks like MobileNet).
**Why it matters**: The findings may not generalize to other pruning methods (e.g., gradient-based pruning, Bayesian pruning, regularization-based pruning), other fault types, or other sensor modalities. The title promises "across neural architectures" but covers only three simple architectures.
**Suggestion**: Expand to at least one additional pruning method (e.g., random pruning as a control, or Taylor-expansion-based pruning). Add a more modern architecture. Acknowledge the scope limitations more prominently in the title or abstract.
**Severity**: Major

### W4: FC architecture is not a meaningful comparator
**Problem**: The FC network has ~691K parameters vs. ~32K (1D-CNN) and ~24K (2D-CNN) — roughly 20-30× larger. Its baseline accuracy (63.6% on CWRU) is far below practical usefulness. The finding that "FC degrades least in absolute terms" (Abstract, p.1) is trivial — when starting from near-chance, there is little accuracy to lose.
**Why it matters**: Including FC as a primary comparator inflates the architecture count but adds little insight. It may mislead readers about "which architecture to choose."
**Suggestion**: Either replace FC with a more competitive lightweight architecture (e.g., a smaller ResNet or a simple 1D-CNN of similar parameter count to the other models), or move FC to supplementary materials and reframe the paper as a comparison of 1D-CNN vs. 2D-CNN only.
**Severity**: Major

### W5: No physically pruned models — zero-masking limitation
**Problem**: The paper uses L1-norm zero-masking rather than physical channel removal (Section 5.3, p.5). As the authors correctly note, this does not yield inference speedup because PyTorch's dense GEMM still runs. However, the paper continues to frame pruning as a "compression strategy" throughout, when zero-masking without structured hardware support is not actually compression.
**Why it matters**: This disconnect between the claimed practical motivation (edge deployment) and the experimental implementation (zero-masking) weakens the paper's applied contributions. The latency measurements merely confirm known PyTorch behavior.
**Suggestion**: Reframe or rename the paper to clarify that this is about pruning robustness (tolerance to weight removal) rather than compression for speed. Add experiments with physically pruned models if acceleration claims are to be made. At minimum, move the practical claims to the limitations section.
**Severity**: Major

---

## Detailed Comments

### Title & Abstract
The title is accurate but somewhat overstated. "Across Neural Architectures" implies broader coverage than three architectures. The abstract is well-structured but includes the FC weakness (low baseline) without adequately caveating it. The claim "prune-then-fine-tune consistently outperformed" (should be "underperformed") — minor typo in logic direction.

### Introduction
Well-motivated. The research question (p.2) is clearly stated. The hypothesis is explicit — this is good practice. However, the motivation for specifically comparing FC, 1D-CNN, and 2D-CNN could be stronger: why these three, and why not include a modern lightweight architecture?

### Methodology
Generally sound. The pruning procedure follows Li et al. (2017) correctly. File-level splitting is a strength. However:
- The training protocol (Section 3.4) does not specify early stopping, batch size, or validation strategy
- The fine-tuning protocol (10 epochs, LR 0.0005) appears chosen arbitrarily

### Results
Tables are clear. The absence of variance measures is a significant weakness. Figure references (Figures 1 and 2) appear in text but the manuscript has no figures embedded — this should be clarified.

### Discussion
The practical recommendations (Section 6.2) are reasonable but should be caveated by the zero-masking limitation. The limitations section (6.3) is honest but understates the fine-tuning budget confound.

### Conclusion
Proportional to the findings. Does not overclaim.

### References
16 references. Adequate but leans heavily on arXiv preprints (4 of 16). Some key pruning theory works are missing (e.g., Frankle & Carbin, "The Lottery Ticket Hypothesis"; Gale et al., "The State of Sparsity").

---

## Questions for Authors

1. Can you provide standard deviations for all reported accuracy values, or alternatively per-seed breakdowns, to allow readers to assess the reliability of the observed differences?
2. The ablation study compares 10-epoch fine-tuning against 50-epoch scratch training. How do the results change when the training budgets are matched?
3. Why was FC included as a primary architecture when its baseline accuracy is far below practical usefulness? Would a lightweight CNN (e.g., a 4-layer 1D-CNN with comparable parameter count) be a more informative comparator?

---

## Minor Issues

### Language / Grammar
- Abstract, line 5: "FC networks, despite low baseline accuracy, degrade least in absolute terms" — consider rewording to clarify that this is an artifact of floor effects
- Section 5.1, p.5: "The hypothesis that 2D-CNN is more pruning-robust is not supported" — consider "was not supported by our experiments" to avoid over-generalizing

### Citation Format
- Reference format is generally consistent. However, Reference 11 (Wang et al., 2025) has a year (2601?) that appears erroneous in the arXiv ID.

### Figures and Tables
- Tables 1 and 2 are clear but would benefit from ± columns showing variability
- Figure references (Figures 1 and 2) are mentioned but no figures are present in the manuscript file
- The paper would benefit from a visualization of the Pareto frontier across architectures

### Layout
- Section numbering is clear and consistent

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 68 | Adequate | First cross-architecture comparison in domain, but methodology is a direct application of Li et al. (2017) |
| Methodological Rigor (25%) | 55 | Weak | Confounded ablation, no variance reporting, limited statistical power, zero-masking vs. physical pruning gap |
| Evidence Sufficiency (25%) | 58 | Weak | 3 seeds, 2 datasets, 1 pruning method — not sufficient for the scope of claims |
| Argument Coherence (15%) | 70 | Adequate | Clear structure, logical flow, but some conclusions outpace the evidence |
| Writing Quality (15%) | 75 | Strong | Clear, professional academic prose |
| **Weighted Average** | **63.4** | | **Major Revision** |

---

---

## Report 2: Peer Reviewer 1 (Methodology)

# Peer Review Report

## Manuscript Information
- **Title**: Structured Pruning Robustness Across Neural Architectures for Bearing Fault Diagnosis
- **Review Date**: July 14, 2026
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
Researcher specializing in experimental design for machine learning benchmarking, model compression evaluation protocols, reproducibility standards, and statistical methodology in engineering applications. Particular focus on avoiding common pitfalls in comparative ML studies.

### Review Focus
Research design rigor, statistical validity of comparisons, reproducibility, ablation methodology, and the integrity of the experimental protocol.

---

## Overall Assessment

### Recommendation
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision**
- [ ] **Reject**

### Confidence Score
5 — Completely within my area of expertise, I am very confident in my assessment.

### Summary Assessment
This paper compares structured pruning robustness across three architectures using a controlled experimental protocol. The use of file-level splitting and multiple seeds represents good methodological practice relative to the current state of the field. The ablation study (prune vs. scratch) is a well-designed experiment in concept. However, I have serious concerns about the statistical characterization of results, a confounding variable in the ablation's training budget, and the absence of key methodological details needed for reproducibility. The scope of the experimental design (one pruning method, one type of sparsity structure) is too narrow to support the paper's broader claims about pruning robustness. I recommend major revision to address these methodological gaps.

---

## Strengths

### S1: File-level splitting prevents data leakage
The paper uses file-level rather than window-level splitting across train/validation/test sets (Section 3.3). This is a known best practice for bearing diagnosis benchmarks, and its adoption here is a clear methodological strength.

### S2: Multi-seed experiments
Running all experiments across 3 random seeds (Section 3.4) is a minimum acceptable practice for stochastic deep learning experiments. The use of fixed seeds [42, 123, 256] enables partial reproducibility.

### S3: Ablation study design
The comparison between prune-then-fine-tune and training from scratch at matched reduced width (Section 5.4) is conceptually well-motivated and directly tests whether pruning preserves useful knowledge beyond what a smaller model could learn independently.

### S4: Reproducibility infrastructure
The companion GitHub repository with full source code, environment specification, and raw results (Section 8) meets or exceeds current reproducibility standards. The explicit "data_manifest" documentation is particularly welcome.

---

## Weaknesses

### W1: Confounded training budget in ablation
**Problem**: The ablation study (Table 2) compares prune-then-fine-tune (10 epochs at LR 0.0005) against training from scratch (50 epochs at LR 0.001 with ReduceLROnPlateau). These differ in both total training steps and learning rate schedule. The scratch model receives 5× the training budget. The observed gap of up to 32 pp could be partly or largely due to insufficient fine-tuning rather than fundamental pruning damage.
**Why it matters**: This is the paper's most important experimental finding. If the confound is not addressed, the conclusion "L1 zero-masking pruning damages model structure" may be incorrect. A more precise statement would be "with 10 epochs of fine-tuning, pruned models underperform models trained from scratch at reduced width."
**Suggestion**: The cleanest fix is to match training budgets: either fine-tune for 50 epochs (the same budget as scratch) or train the scratch model for only 10 epochs. Alternatively, show that the gap persists when fine-tuning is extended to 50 epochs. If resource constraints prevent re-running, explicitly acknowledge this confound as a major limitation.
**Severity**: Critical

### W2: No variance reporting
**Problem**: Tables 1 and 2 report only mean accuracy across 3 seeds with no standard deviations, confidence intervals, or per-seed values. With n=3, individual seed results can vary substantially in deep learning experiments.
**Why it matters**: Without variance information, readers cannot assess: (a) whether the ranking of architectures is statistically reliable, (b) whether the 20.2 pp vs. 27.2 pp difference between 1D-CNN and 2D-CNN at 75% pruning on CWRU is meaningful, or (c) whether the ablation gaps are consistent across seeds.
**Suggestion**: Report full per-seed results or at minimum standard deviations. For the architecture comparison, compute bootstrap confidence intervals on the accuracy drop from 0% to 75% pruning. Increase the number of seeds — 5-10 is typical for this type of study.
**Severity**: Critical

### W3: Hypothesis testing without statistical framework
**Problem**: The paper's central research question is comparative: "How do FC, 1D-CNN, and 2D-CNN compare in robustness to structured pruning?" (Section 1). The hypothesis about 2D-CNN is evaluated by inspecting the accuracy drop magnitudes (Section 5.1), but no statistical test is applied.
**Why it matters**: The conclusion "2D-CNN is not more pruning-robust" is an inference from a difference of 7 percentage points in accuracy drop across two architectures, measured with 3 seeds each. This difference may or may not be statistically significant.
**Suggestion**: Apply a formal test. For example, compute the accuracy drop per seed per architecture and use a paired t-test or Mann-Whitney U test to compare the distributions. Alternatively, report bootstrap confidence intervals on the difference in accuracy drops. Even a simple effect size (Cohen's d) would be informative.
**Severity**: Major

### W4: Insufficient detail on training protocols
**Problem**: Several important methodological details are missing. Batch size is not reported. The validation strategy (how the best model is selected across 50 epochs) is not specified. Early stopping is not mentioned. The "ReduceLROnPlateau" scheduler parameters are not given (patience, factor, threshold). The STFT parameters for 2D-CNN (window length, hop length, FFT size, window function) are not provided.
**Why it matters**: These details are essential for reproducibility. Without them, another research group cannot replicate the experiments. Some of these choices (e.g., STFT parameters) affect the 2D-CNN's input representation and could influence pruning robustness.
**Suggestion**: Add a comprehensive table of training hyperparameters (batch size, learning rate schedule details, validation metric, early stopping criteria, weight initialization). Provide STFT parameters for spectrogram generation. A full hyperparameter section in the supplementary materials would be appropriate.
**Severity**: Major

### W5: Single pruning method limits generalizability of claims
**Problem**: The paper tests only L1-norm structured channel pruning (Li et al., 2017) with zero-masking. It does not include: (a) random pruning as a baseline control, (b) any other structured pruning criterion (e.g., Taylor expansion, gradient-based, entropy-based), or (c) unstructured pruning.
**Why it matters**: The paper's title and framing suggest general conclusions about "pruning robustness" across architectures, but these conclusions may be specific to L1-norm structured pruning. The L1 criterion is known to work well for some architectures and poorly for others, so the observed ranking could change with a different pruning criterion.
**Suggestion**: Add at minimum a random pruning control to distinguish architecture-specific pruning robustness from generic parameter redundancy. Ideally, add one more pruning criterion (e.g., Taylor-first-order or FPGM). Revise the title and claims to specify "L1-norm structured pruning robustness."
**Severity**: Major

---

## Detailed Comments

### Title & Abstract
The abstract correctly summarizes the main findings. However, the FC result ("degrade least in absolute terms") should be caveated as a floor effect. The claim about prune-then-fine-tune being "consistently inferior" needs the training budget confound addressed.

### Methodology
- Section 3.1: "Pruning is applied post-training, followed by fine-tuning for 10 epochs" — why 10? Was this based on convergence analysis?
- Section 3.2: The FC network is an order of magnitude larger than the CNNs. This parameter mismatch confounds the architecture comparison with model capacity effects. A controlled comparison would match parameter counts.
- Section 3.4: Batch size? Weight initialization scheme? These are missing.

### Results
- The absence of standard deviations in Tables 1 and 2 is the single biggest methodological weakness.
- Section 5.3: The latency analysis is informative but the conclusion ("true speedup requires hardware-aware pruning") makes the zero-masking approach seem less useful for deployment than the paper's framing suggests.

### Discussion
The limitations section (6.3) acknowledges the zero-masking issue and the small number of seeds, but does not mention the fine-tuning budget confound. This should be added.

### References
Missing: Frankle & Carbin (2019) "The Lottery Ticket Hypothesis" — directly relevant to whether pruning finds good subnetworks. Missing: A systematic discussion of why L1 pruning might fail that connects to the broader pruning literature.

---

## Questions for Authors

1. Can you provide standard deviations or per-seed results for all experiments? How many seeds would be needed for the observed differences between architectures to be statistically significant?
2. If you extend the fine-tuning from 10 to 50 epochs (matching the scratch model's training budget), does the prune-then-fine-tune gap persist or shrink? This is critical for the paper's central claim.
3. What are the complete training hyperparameters (batch size, validation strategy, scheduler parameters, initialization)? Can you provide these in a supplementary hyperparameter table?
4. Did you test whether random pruning produces the same architecture ranking? If L1 selection is not better than random for these architectures, that would be an important finding.

---

## Minor Issues

### Language / Grammar
- Section 5.4: "pruning disrupts the collaborative representation" — this is an interpretation, not an observation. Consider "consistent with the hypothesis that pruning disrupts..."

### Methodology Details
- STFT parameters (window size, hop length, FFT size) for the 2D-CNN spectrograms are not specified anywhere. These are essential for reproducibility.
- The "file-level splitting" procedure described in Section 3.3 needs more detail: how exactly are files assigned to train/val/test? Stratified by fault type?

### Figures and Tables
- Table 2: The FC "Prune+FT" and "Scratch" values for CWRU at 75% are 56.53 and 55.23 (Gap +1.30), while for MFPT they are both 56.73 (Gap 0.00). The MFPT values being identical across all four decimal places is suspicious — please verify these numbers.

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 70 | Adequate | Novel comparison, but methodology is standard |
| Methodological Rigor (25%) | 52 | Weak | Confounded ablation, no variance reports, missing details |
| Evidence Sufficiency (25%) | 50 | Weak | 3 seeds, 1 pruning method, no confidence intervals |
| Argument Coherence (15%) | 68 | Adequate | Generally logical but some conclusions overreach evidence |
| Writing Quality (15%) | 72 | Adequate | Clear but some precision issues in methodology description |
| **Weighted Average** | **60.4** | | **Major Revision** |

---

---

## Report 3: Peer Reviewer 2 (Domain Expert)

# Peer Review Report

## Manuscript Information
- **Title**: Structured Pruning Robustness Across Neural Architectures for Bearing Fault Diagnosis
- **Review Date**: July 14, 2026
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Domain expert in bearing fault diagnosis, vibration signal processing, and condition monitoring. Deep familiarity with CWRU and MFPT datasets, their limitations, and the trajectory of deep learning methods in this field. Published work on benchmarking methodologies for fault diagnosis.

### Review Focus
Literature coverage, theoretical framework appropriateness for bearing diagnosis, dataset choices, practical relevance of findings to the condition monitoring community, and accuracy of domain-specific claims.

---

## Overall Assessment

### Recommendation
- [ ] **Accept**
- [ ] **Minor Revision**
- [x] **Major Revision**
- [ ] **Reject**

### Confidence Score
5 — Completely within my area of expertise, I am very confident in my assessment.

### Summary Assessment
This paper addresses a practically relevant question for the bearing fault diagnosis community: how do different neural architectures tolerate structured pruning when preparing models for edge deployment? The cross-architecture comparison is novel and fills a real gap. The use of file-level splitting and the honest reporting of a falsified hypothesis are commendable. However, I have concerns about the literature coverage (several important recent works are missing), the choice of datasets (both are clean lab data — not representative of real deployment), and some domain-specific technical claims that require clarification. The comparison between architectures of vastly different parameter counts (FC at 691K vs. CNNs at 24-32K) reduces the practical relevance from a domain perspective, since no practitioner would choose an FC network for vibration diagnosis today. The paper would benefit from deeper engagement with the bearing diagnosis literature on lightweight model design. I recommend major revision.

---

## Strengths

### S1: Addresses a real practical gap
The bearing diagnosis community increasingly needs guidance on model compression for edge deployment. The question "which architecture should I choose if I need to compress it?" is directly relevant to practitioners designing embedded monitoring systems. The paper's practical recommendations (Section 6.2) are useful starting points.

### S2: Methodologically sound dataset handling
File-level splitting (Section 3.3) correctly addresses the data leakage issue documented by Xu et al. (2022), which remains a widespread problem in the bearing diagnosis literature. The paper also reports dataset documentation cards (Section 8), which is best practice.

### S3: Falsified hypothesis reported honestly
The hypothesis that 2D-CNN would be more pruning-robust (Section 1) is clearly stated and then honestly reported as unsupported by the evidence (Section 5.1). In a field where null results are rarely published, this is a valuable contribution.

### S4: Ablation provides actionable guidance
The finding that training from scratch at reduced width outperforms prune-then-fine-tune (Section 5.4) has direct, actionable implications: if you need a smaller model, just train one rather than pruning a large one.

---

## Weaknesses

### W1: Limited literature coverage — missing recent lightweight diagnosis works
**Problem**: The related work section (2.2) cites 3 compression papers (Guo et al., 2024; Wang et al., 2025; Chen et al., 2023) but misses several important lightweight diagnosis architectures published in 2024-2025, such as MobileNet-inspired vibration networks, depthwise-separable convolution approaches for bearing diagnosis, and transformer pruning studies. Notably, the paper does not cite works on pruning for 1D-CNNs specifically in fault diagnosis.
**Why it matters**: Without engaging with the full landscape of lightweight diagnosis models, the paper's claim to be the "first systematic comparison" may be contested, and readers lack context for where pruning fits among other compression approaches.
**Suggestion**: Add a broader literature search covering: (a) lightweight CNN designs for vibration diagnosis, (b) knowledge distillation approaches beyond DKDL-Net, (c) any prior work on pruning for 1D-CNNs in time-series classification, (d) TinyML applications in condition monitoring.
**Severity**: Major

### W2: CWRU and MFPT are both clean laboratory data
**Problem**: Both datasets are collected under controlled laboratory conditions. CWRU is a single bearing type under controlled loads; MFPT is similarly lab-based. Real-world bearing diagnosis involves speed variation, load cycling, multiple fault types simultaneously, and environmental noise that differs from additive Gaussian. The paper tests only additive Gaussian noise (Section 5.2) at two SNR levels.
**Why it matters**: The practical recommendations (Section 6.2) are aimed at "practitioners deploying...under compression constraints," but the experiments use data that may not reflect real deployment conditions. Pruning robustness may behave differently under domain shift, speed variation, or multi-fault conditions.
**Suggestion**: Add at minimum one real-world or more challenging dataset (e.g., XJTU-SY, Paderborn University dataset, or a dataset with speed variation). Alternatively, add experiments with more realistic noise profiles (speed variation, industrial noise recordings rather than additive Gaussian). Clearly state the scope limitation regarding laboratory vs. real-world data.
**Severity**: Major

### W3: FC architecture is not practically relevant for bearing diagnosis
**Problem**: The FC network (691K params, 63.6% accuracy on CWRU) is not a competitive architecture for bearing fault diagnosis. Modern lightweight CNNs achieve 95%+ accuracy with far fewer parameters. Including FC inflates the architecture comparison without providing useful guidance.
**Why it matters**: From a domain perspective, no practitioner would deploy a pure FC network for bearing fault classification in 2026. The finding that "FC degrades least" is a floor effect (near-chance baseline) and could mislead readers.
**Suggestion**: Replace FC with a competitive lightweight architecture (e.g., a 4-layer 1D-CNN with comparable parameter count to the 2D-CNN, or a modern lightweight network like a simplified MobileNetV1 for 1D signals). If FC is retained, move it to supplementary material and explicitly warn readers about the floor effect.
**Severity**: Major

### W4: Evaluation on CWRU with only 4 classes is a simplified setting
**Problem**: The CWRU experiments use 4 classes (normal, ball fault, inner race, outer race) at a single fault diameter (0.007"). Real CWRU benchmarks typically include more fault types and multiple severity levels (0.007", 0.014", 0.021").
**Why it matters**: Pruning robustness may differ when the model must discriminate more fine-grained fault categories (same fault type, different severity). The finding of "1D-CNN is more pruning-robust" might not generalize to the full multi-class, multi-severity setting.
**Suggestion**: Either expand to the full CWRU classification task (10+ classes including different fault diameters and positions) or explicitly limit the claim to the 4-class setting used.
**Severity**: Major

### W5: Missing discussion of STFT preprocessing burden for 2D-CNN
**Problem**: The paper compares 2D-CNN (using STFT spectrograms) against 1D-CNN (using raw waveforms) and FC (using 1024-point segments). The computational and memory cost of STFT preprocessing for 2D-CNN is not discussed anywhere.
**Why it matters**: For edge deployment, preprocessing overhead matters. If 1D-CNN achieves comparable or better pruning robustness without needing STFT, this strengthens the case for 1D-CNN. The absence of this discussion makes the comparison incomplete.
**Suggestion**: Add a discussion (or ideally measurements) of: (a) STFT computation time per sample, (b) additional memory for storing spectrograms, (c) whether STFT can be efficiently implemented on edge hardware.
**Severity**: Minor

---

## Detailed Comments

### Title & Abstract
Accurate and clear. The FC caveat should be more prominent in the abstract.

### Introduction
Well-motivated. The grounding in vibration-based diagnosis is appropriate. However, the claim that "deep learning has become the dominant approach" could cite more recent comprehensive surveys (e.g., from 2023-2025).

### Related Work
- Section 2.1: The CWRU discussion should cite the Smith & Randall (2015) benchmark paper correctly — it is already cited, good.
- Section 2.2: Missing discussion of:
  - Pruning specifically for 1D time-series models
  - Lightweight architecture search for bearing diagnosis
  - Prior work on comparing architectures for bearing diagnosis (without pruning)

### Results
- The CWRU accuracy values (Table 1): 2D-CNN at 95.6% unpruned is reasonable but lower than some reported results (97-99%). This may reflect the harder file-level split. This is actually a strength — the results are honest.
- The MFPT results are surprising: 1D-CNN at 99.9% unpruned but 2D-CNN at only 89.8%. This large discrepancy needs more explanation. Is MFPT's signal structure particularly suited to raw-waveform processing? This could be discussed more.

### Discussion
The practical recommendations are sensible. However, they should be qualified by the limited experimental scope. The recommendation "avoid L1-structured pruning with zero-masking" is the strongest practical takeaway and is appropriately emphasized.

### Conclusion
Appropriate in scope and tone.

---

## Questions for Authors

1. Can you comment on why 2D-CNN performs so much worse on MFPT (89.8%) than 1D-CNN (99.9%)? Is this related to MFPT's different sampling characteristics or signal structure?
2. How does the pruning robustness ranking change if you include a more modern lightweight architecture (e.g., depthwise-separable CNN, or a MobileNet-style network for vibration data)?
3. Have you considered validating the pruning robustness findings on a more challenging dataset with real-world noise conditions (e.g., XJTU-SY or Paderborn)?

---

## Minor Issues

### Language / Grammar
- Section 2.1: "The CWRU bearing dataset...has become the de facto standard benchmark" — consider adding a nuance that over-reliance on CWRU is a known issue in the field.

### Dataset Details
- Table 1 caption: Note that dataset splits are file-level, not random-window-level. This should be stated in the caption for clarity.
- Section 3.3: The resampling of MFPT from 97.6 kHz to 12 kHz is mentioned but details (resampling method, anti-aliasing filter) are not provided.

### Domain Terminology
- "Vibration segment" (Section 3.2): The segment length of 1024 points at 12 kHz represents ~85 ms of vibration. This is worth noting — is this long enough for low-speed bearing fault signatures?
- Consider discussing the bearing characteristic frequencies and whether the segment length captures multiple rotations of the shaft at the tested load conditions.

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 72 | Adequate | Novel cross-architecture comparison in this domain |
| Methodological Rigor (25%) | 58 | Weak | Good dataset handling but narrow scope limits rigor |
| Evidence Sufficiency (25%) | 52 | Weak | 2 lab datasets insufficient to support deployment recommendations |
| Argument Coherence (15%) | 70 | Adequate | Clear logic but some domain-specific gaps |
| Writing Quality (15%) | 73 | Strong | Clear domain-appropriate language |
| Literature Integration (optional) | 60 | Adequate | Missing several recent lightweight diagnosis works |
| **Weighted Average** | **63.1** | | **Major Revision** |

---

---

## Report 4: Peer Reviewer 3 (Perspective / Cross-Disciplinary)

# Peer Review Report

## Manuscript Information
- **Title**: Structured Pruning Robustness Across Neural Architectures for Bearing Fault Diagnosis
- **Review Date**: July 14, 2026
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Cross-disciplinary researcher from the efficient ML / model compression community. Expertise in neural network pruning theory, sparse training, quantization, and edge deployment of deep learning models across application domains (CV, NLP, time series). Brings perspective on how this work connects to the broader pruning literature and what general lessons can be learned.

### Review Focus
Connection to broader pruning theory, practical implications for edge deployment, cross-disciplinary borrowing opportunities, validity of pruning methodology choices, and the extent to which findings from this domain-specific study inform the general understanding of pruning robustness.

---

## Overall Assessment

### Recommendation
- [ ] **Accept**
- [x] **Minor Revision**
- [ ] **Major Revision**
- [ ] **Reject**

### Confidence Score
4 — Mostly within my area of expertise, high confidence.

### Summary Assessment
From the perspective of the broader model compression community, this paper makes a modest but useful contribution: it demonstrates that pruning robustness is architecture-dependent in the bearing diagnosis domain, and it provides evidence that L1 zero-masking pruning adds little value over training a smaller model from scratch. The latter finding is consistent with observations in other domains (Renda et al., 2020, "Comparing Rewinding and Fine-Tuning in Neural Network Pruning") and contributes to the growing understanding that learned weights are not as important for pruning outcomes as the training process itself. However, the paper's narrow scope (one pruning method, simple architectures) limits what it can tell us about pruning robustness more generally. The lack of physically structured pruning (actual channel removal) means the latency and efficiency claims are not connected to the pruning mechanism used. I believe the paper is publishable after minor revisions to better frame its contribution within the broader pruning landscape and to tighten the connection between pruning methodology and deployment claims.

---

## Strengths

### S1: Cross-architecture pruning comparison is underexplored
Most pruning papers optimize one architecture in isolation. This paper's comparative approach is unusual and valuable. It asks "which architecture is more pruning-friendly?" rather than "how do I prune this particular architecture?" — a question more practitioners should consider.

### S2: Ablation connects to a general pruning debate
The finding that training from scratch outperforms prune-then-fine-tune (Section 5.4) echoes findings from the broader pruning literature. Renda et al. (2020) showed that weight rewinding (resetting to early-training weights before fine-tuning) outperforms straightforward fine-tuning after pruning. Liu et al. (2019, "Rethinking the Value of Network Pruning") famously found that pruned architectures trained from scratch match or exceed pruned-and-fine-tuned models. The paper's ablation aligns with this literature, which the authors could leverage more explicitly.

### S3: Honest limitations section
The limitations enumerated in Section 6.3 — zero-masking, limited seeds, simple noise model — are openly acknowledged. This is good scientific practice. The latency measurements (Section 5.3) correctly identify that zero-masking does not yield speedup with dense operations, which many pruning papers fail to acknowledge.

### S4: Reproducibility commitment
The detailed reproducibility statement (Section 8) is a model for other papers in this space.

---

## Weaknesses

### W1: Disconnect between pruning framing and deployment claims
**Problem**: The paper is framed around edge deployment (Abstract, Introduction), but uses L1 zero-masking that does not reduce model size or inference latency on standard hardware. As Section 5.3 correctly shows, PyTorch's dense GEMM operations do not benefit from sparsity introduced by zero-masking. The paper even acknowledges that "true speedup requires hardware-aware pruning" (Section 5.3).
**Why it matters**: This creates a fundamental tension: the paper motivates pruning as a compression technique, but the experiments do not compress the model in any meaningful way (no parameter reduction, no latency reduction). This disconnect is confusing for readers and weakens the paper's practical contribution.
**Suggestion**: Restructure the paper around "pruning robustness" (tolerance to parameter removal) rather than "model compression for edge deployment." The latency analysis (Section 5.3) should be moved to limitations or reframed as a cautionary note. Alternatively, add experiments with physically pruned models (actual channel removal in network architecture).
**Severity**: Critical

### W2: Missing connection to the general pruning literature
**Problem**: The related work (Section 2.2) is limited to bearing diagnosis compression papers. It does not engage with the broader pruning literature that is directly relevant. Key missing works include: (a) Liu et al. (2019) "Rethinking the Value of Network Pruning" — argues that pruned architectures, not weights, matter; (b) Frankle & Carbin (2019) "Lottery Ticket Hypothesis" — directly relevant to whether pruning preserves useful structure; (c) Renda et al. (2020) "Comparing Rewinding and Fine-Tuning" — directly addresses the fine-tuning regime; (d) Gale et al. (2019) "The State of Sparsity" — benchmarks pruning methods.
**Why it matters**: The paper's most interesting result (scratch beats prune-then-fine-tune) is well-known in the general pruning community but presented as a novel finding. Engaging with this literature would strengthen the paper and help readers understand where this contribution fits.
**Suggestion**: Add a paragraph connecting the ablation finding to Liu et al. (2019) and Renda et al. (2020). Explicitly state whether the findings confirm or contradict general pruning theory in the bearing diagnosis context.
**Severity**: Major

### W3: Single pruning criterion limits architectural insight
**Problem**: The paper uses only L1-norm magnitude pruning (Li et al., 2017). This criterion is known to have limitations — it assumes channel importance is proportional to weight magnitude, which is not always true (e.g., a channel with large weights that are all redundant vs. a channel with small but critical weights). Alternative criteria (Taylor expansion, gradient*weight product, or even random pruning as a control) would provide a more complete picture.
**Why it matters**: If the architecture ranking is consistent across pruning criteria, the finding is robust and general. If the ranking changes, the paper's conclusions about "1D-CNN is more pruning-robust" are criterion-specific. The paper cannot distinguish these possibilities with only one criterion.
**Suggestion**: Add at minimum a random pruning baseline to test whether L1 selection actually adds value over random removal. If L1 ≈ random for some architectures, that is an important finding. Ideally, add one more criterion (e.g., Taylor-first-order approximation).
**Severity**: Major

### W4: No analysis of which channels / features are actually removed
**Problem**: The paper reports only aggregate accuracy after pruning. There is no analysis of: (a) which types of channels are pruned (early vs. late layers), (b) whether pruning removes redundant channels or critical ones, (c) how the pruning pattern differs across architectures, (d) whether the model can recover from pruning of different layers.
**Why it matters**: Understanding *why* 2D-CNN is less robust requires per-layer analysis. Is it because the first convolutional layer (16 channels) has too few channels to tolerate pruning? Is it because spectrogram features are more entangled? Without this analysis, the explanation (Section 6.1) is speculative.
**Suggestion**: Add per-layer pruning sensitivity analysis or per-layer accuracy after pruning. At minimum, show the layer-wise pruning ratios and their effect on accuracy.
**Severity**: Major

### W5: Superficial noise robustness analysis
**Problem**: The noise robustness section (5.2) reports accuracy drops at two SNR levels with minimal analysis. There is no comparison of noise robustness between pruned and unpruned models at matched parameter counts (the scratch model from the ablation would be an excellent control). There is no discussion of whether pruning affects noise robustness differently across architectures.
**Why it matters**: The noise robustness results are potentially the most practically relevant (edge sensors are noisy). The current analysis is too brief to be useful.
**Suggestion**: Compare noise robustness of pruned models against scratch models at matched width. Show per-architecture noise robustness trends as a function of pruning ratio. Consider more realistic noise types (e.g., industrial noise recordings, impulse noise).
**Severity**: Minor

---

## Detailed Comments

### Title & Abstract
The abstract is well-written but overstates the practical implications given the zero-masking disconnect. The FC floor effect should be more prominently caveated.

### Introduction
The research question and hypothesis are clearly stated. The motivation for the architecture choice is reasonable but would benefit from a broader ML-systems perspective — why not include a transformer-based architecture for completeness?

### Related Work
The narrow literature coverage is the biggest weakness from my perspective. The paper should engage with Liu et al. (2019), Frankle & Carbin (2019), Renda et al. (2020), and Gale et al. (2019) to properly contextualize the findings.

### Methodology
The pruning protocol (Section 3.1) is standard but should specify: (a) whether batch normalization layers are pruned (or their channels aligned with pruned conv channels), (b) whether the pruning is applied globally or per-layer, (c) the exact zero-masking implementation.

### Results
- The ablation results (Table 2) are the most interesting part. The gap for 1D-CNN on CWRU (−32.05 pp) is dramatic and deserves more discussion.
- The latency measurements (Section 5.3) confirm known behavior. Consider reframing as a negative result: "zero-masking does not yield speedup, emphasizing the need for hardware-aware pruning."

### Discussion
The practical recommendations (Section 6.2) are sensible but should be connected to the general pruning literature recommendation: if compression is needed, train a smaller model rather than pruning a larger one.

---

## Questions for Authors

1. Your finding that training from scratch at reduced width outperforms prune-then-fine-tune aligns with Liu et al. (2019). Can you discuss how your results compare with or extend their findings in the bearing diagnosis context?
2. Did you observe different pruning patterns across layers (early vs. late) for different architectures? If so, how do these patterns relate to the observed accuracy drops?
3. Have you considered testing random pruning as a baseline? If L1 pruning does not outperform random pruning for these architectures, that would be a significant finding.

---

## Minor Issues

### Terminology
- "Structured pruning" should be explicitly distinguished from "unstructured pruning" early in the paper. A brief definition would help readers not deeply familiar with pruning taxonomies.
- "L1 zero-masking pruning" might be more precisely called "L1-norm-based magnitude pruning with weight zeroing" to avoid confusion with actual structured channel removal.

### Broader Impact
- Consider a brief discussion of the environmental implications: if training smaller models from scratch is better than pruning, what does this mean for the carbon footprint of ML-based diagnosis research?

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 70 | Adequate | Cross-architecture comparison is novel within domain |
| Methodological Rigor (25%) | 60 | Adequate | Sound basic design but zero-masking issues and missing controls |
| Evidence Sufficiency (25%) | 58 | Weak | Single pruning criterion limits general claims |
| Argument Coherence (15%) | 72 | Adequate | Clear but the framing-deployment disconnect weakens coherence |
| Writing Quality (15%) | 75 | Strong | Well-written academic prose |
| Significance & Impact (optional) | 68 | Adequate | Useful for practitioners but limited generalizability |
| **Weighted Average** | **65.1** | | **Minor Revision** |

---

---

## Report 5: Devil's Advocate (DA) Review

# Peer Review Report — Devil's Advocate

## Manuscript Information
- **Title**: Structured Pruning Robustness Across Neural Architectures for Bearing Fault Diagnosis
- **Review Date**: July 14, 2026
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Devil's Advocate Reviewer

### Reviewer Identity
Critical analyst specialized in stress-testing core arguments, detecting logical fallacies, confirmation bias, and overgeneralization. The role is not to oppose for opposition's sake, but to identify the strongest counter-arguments to the paper's claims and help the authors strengthen their reasoning.

### Review Focus
Core argument validation, cherry-picking detection, logic chain validation, alternative interpretations, overgeneralization risks, and the "so what?" test.

---

## Overall Assessment

### Recommendation
The Devil's Advocate does not make a publication recommendation per the skill protocol. The role is to surface critical challenges to the paper's core arguments.

### Confidence Score
5 — Completely within my area of expertise.

### Summary Assessment
The Devil's Advocate identifies **three CRITICAL challenges** to the paper's core argument chain, plus several major concerns. The first CRITICAL challenge is the confounded ablation comparison (10-epoch fine-tune vs. 50-epoch scratch), which undermines the paper's strongest claim. The second CRITICAL challenge is that the paper over-interprets the falsification of its 2D-CNN hypothesis — the data are consistent with multiple alternative explanations. The third CRITICAL challenge is the framing-deployment disconnect: the paper claims practical relevance for edge deployment but uses zero-masking that cannot actually compress models. The "so what?" question — what should a practitioner do differently after reading this paper — does not have a clear answer that is supported by the evidence.

---

## Strongest Counter-Argument

The paper's central claim is: "L1-norm structured pruning via zero-masking damages model structure, and training from scratch at reduced width is consistently superior" (Section 5.4, p.6). The strongest counter-argument is that the experimental design cannot distinguish between "pruning damages model structure" and "insufficient fine-tuning after pruning." The fine-tuning protocol (10 epochs at LR 0.0005 with no scheduling) is a fraction of the initial training budget (50 epochs at LR 0.001 with ReduceLROnPlateau). The scratch model receives 5× the optimizer steps with an adaptive learning rate. The observed 32 pp gap on CWRU 1D-CNN may reflect training budget disparity rather than inherent pruning damage. In the pruning literature, Renda et al. (2020) showed that longer fine-tuning (especially with learning rate rewinding) substantially closes the gap between pruned and scratch models. Without controlling for total training budget, the paper's central conclusion is not supported by the evidence.

Furthermore, even if the gap persists with matched budgets, the paper's framing presents this as a novel finding, but Liu et al. (2019) already demonstrated this for large-scale image classification. The incremental contribution is therefore limited to demonstrating that the same phenomenon occurs in bearing diagnosis — a useful but modest extension.

---

## Issue List

### CRITICAL Issues

#### DA-C1: Confounded ablation confounds the core claim
**Dimension**: Methodological Validity
**Location**: Section 5.4, Table 2; also referenced in Abstract, Discussion (6.2), and Conclusion
**Problem**: The ablation compares prune-then-fine-tune (10 epochs, fixed LR 0.0005) vs. training from scratch (50 epochs, adaptive LR). This confound alone could produce the observed gap. The paper presents the gap as evidence that "pruning damages model structure" — but insufficient training is a simpler, more parsimonious explanation.
**Why CRITICAL**: This is the paper's primary novel contribution. If the confound is not addressed, the conclusion lacks empirical support.

#### DA-C2: The falsified hypothesis has multiple interpretations
**Dimension**: Logical Inference / Cherry-Picking
**Location**: Section 5.1, p.5; Section 6.1
**Problem**: The paper's hypothesis that "2D-CNN is more pruning-robust" was falsified (27.2 pp drop vs. 20.2 pp for 1D-CNN at 75% pruning on CWRU). The paper attributes this to "distributed spectrogram features" and "fewer total channels" (Section 6.1). However, alternative explanations are equally or more plausible: (a) 2D-CNN has only 16 initial channels vs. 32 for 1D-CNN — pruning 75% of 16 channels leaves only 4 active channels, which may be too few regardless of feature distribution; (b) the STFT spectrograms for MFPT are lower quality (resampled from 97.6 kHz to 12 kHz), disadvantaging 2D-CNN on that dataset; (c) the 2D-CNN may be undertrained relative to its capacity (lower accuracy on MFPT even unpruned: 89.8% vs. 99.9% for 1D-CNN). The paper does not control for these confounds.
**Why CRITICAL**: The paper's central hypothesis test is inconclusive because the architecture comparison includes multiple uncontrolled variables (channel count, input representation quality, dataset-specific suitability).

#### DA-C3: Framing-deployment disconnect makes practical claims unsupported
**Dimension**: Argument Coherence / "So What?" Test
**Location**: Introduction (p.1), Abstract (line 1-2), Discussion (Section 6.2)
**Problem**: The paper frames itself around edge deployment but uses zero-masking pruning that produces no model size reduction, no parameter reduction, and no latency speedup on standard hardware. The paper acknowledges this (Section 5.3) but the Abstract and Introduction continue to frame pruning as a "compression strategy." The practical recommendations (Section 6.2) urge practitioners to "avoid L1-structured pruning with zero-masking" — but the evidence only shows it doesn't help with accuracy retention, not that it fails as a compression method (since it isn't one).
**Why CRITICAL**: The paper's motivation and its experimental implementation are misaligned. This undermines the entire applied contribution.

---

### MAJOR Issues

#### DA-M1: FC architecture inclusion inflates apparent coverage
**Dimension**: Overgeneralization
**Location**: Section 3.2, Results passim
**Problem**: The paper claims to compare "three architectures" but FC at 691K params and near-chance accuracy is not a meaningful comparator. Its inclusion gives the appearance of broader coverage while adding no useful insight. The claim that "FC degrades least" (Abstract) is mathematically trivial given its low starting point.
**Suggestion**: Remove FC from primary comparisons or explicitly warn about floor effects.

#### DA-M2: Alternative interpretation of the ablation — architectural lottery
**Dimension**: Ignored Alternative Explanation
**Location**: Section 5.4
**Problem**: The ablation finding (scratch beats prune-then-fine-tune) could be explained by the "lottery ticket hypothesis" (Frankle & Carbin, 2019): training from scratch at reduced width finds a different, potentially better subnetwork than pruning a pretrained large network. The paper does not consider this interpretation.
**Suggestion**: Discuss the lottery ticket hypothesis as an alternative explanation. Test whether the pruned subnetwork (before fine-tuning) would have performed well if trained from that initialization.

#### DA-M3: The "moderate compression" recommendation is not tested
**Dimension**: Overgeneralization
**Location**: Section 6.2, Recommendation 2
**Problem**: The paper recommends 1D-CNN for "moderate compression (≤50%)" based on CWRU results (90.8% → 77.6%). However, 77.6% accuracy on a 4-class balanced dataset may not be "acceptable" for real deployment. The recommended threshold of ≤50% pruning is not tested against a required accuracy floor.
**Suggestion**: Define an explicit accuracy threshold for "acceptable" deployment and test at which pruning ratio each architecture crosses it.

#### DA-M4: No evidence that L1 selection beats random
**Dimension**: Confirmation Bias / Missing Control
**Location**: Section 4, Section 5
**Problem**: Section 4 validates that L1 pruning targets low-magnitude channels (mean norm 2.48 vs. 5.79 for random). But this does not demonstrate that L1-selected channels are less important for accuracy. Without comparing L1 pruning accuracy against random pruning accuracy, the paper cannot claim that L1 selection is meaningful.
**Suggestion**: Add a random pruning baseline. If L1 pruning and random pruning produce similar accuracy drops, the paper's findings about "pruning robustness" are actually about "random parameter removal robustness."

#### DA-M5: Noise robustness analysis does not connect to pruning claims
**Dimension**: Missing Analysis
**Location**: Section 5.2
**Problem**: The noise robustness results are reported but not analyzed in the context of pruning. The key question is: does pruning change noise robustness? A model that achieves 70% accuracy clean but 30% under noise is less useful than one that achieves 65% clean but 60% under noise. Pruning may differentially affect noise robustness across architectures.
**Suggestion**: Compare noise robustness as a function of pruning ratio for each architecture. Include the scratch ablation models as a control to separate pruning damage from size effects.

---

## Ignored Alternative Explanations/Paths

1. **Alternative Explanation A — Channel count confound**: 2D-CNN has fewer channels per layer (16-32-64) than 1D-CNN (32-64-128). The difference in pruning robustness may be entirely explained by channel count rather than "distributed features." Pruning 75% of 16 channels = 4 remaining channels, which is below any reasonable minimum for a convolutional layer.

2. **Alternative Explanation B — Input representation quality**: The 2D-CNN operates on STFT spectrograms. On CWRU (12 kHz native), STFT quality may be adequate; on MFPT (resampled to 12 kHz from 97.6 kHz), the spectrograms may be lower quality due to resampling artifacts. This could explain why 2D-CNN performs worse on MFPT (89.8%) than 1D-CNN (99.9%).

3. **Alternative Explanation C — Pruning criterion mismatch**: L1-norm magnitude pruning is designed for conv layers where weight magnitude relates to feature importance. For FC layers, the relationship between weight magnitude and importance is less clear. The FC network's apparent "robustness" may reflect the ineffectiveness of L1 pruning for FC layers rather than actual architecture-level robustness.

4. **Alternative Path D — Quantization instead of pruning**: The paper concludes that pruning is ineffective, but does not consider alternatives within the compression toolkit. Quantization-aware training or knowledge distillation may be more suitable for these architectures and datasets. The recommendation should note these alternatives.

---

## Missing Stakeholder Perspectives

1. **Edge hardware engineers**: The paper's zero-masking approach is exactly what edge hardware engineers advised *against* in recent TinyML literature. A hardware-aware perspective would emphasize that pruning must be structured (actual channel removal) to yield benefits on MCU-class devices.
2. **Maintenance practitioners**: The paper assumes that accuracy alone determines deployment suitability. Practitioners also care about false positive rates (missed faults are costly), inference reliability under varying conditions, and model update frequency. None of these are considered.
3. **Dataset curators**: The paper identifies limitations of CWRU and MFPT but does not provide specific guidance for what a better benchmark dataset would look like for pruning robustness studies.

---

## Observations (Non-Defects)

1. The paper's structure is clear and well-organized. Tables are readable.
2. The AI-use disclosure (Section 9) is transparent and appropriately scoped.
3. The reproducibility commitment is commendable and should be standard practice.
4. The honest reporting of a falsified hypothesis, even if the interpretation has issues, is scientifically admirable.
5. The finding about latency (Section 5.3) is actually an important negative result that many pruning papers overlook.
6. The paper would benefit from a visualization showing the accuracy-parameter Pareto frontier by architecture and pruning ratio.

---

---

# Phase 2 — Editorial Synthesis & Decision

---

# Editorial Decision

## Manuscript Information
- **Title**: Structured Pruning Robustness Across Neural Architectures for Bearing Fault Diagnosis
- **Submission Date**: N/A (working paper)
- **Decision Date**: July 14, 2026
- **Review Round**: Round 1

---

## Decision

### Major Revision

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC | Editor-in-Chief (MSSP) | Major Revision | 5 |
| Reviewer 1 | Methodology Expert | Major Revision | 5 |
| Reviewer 2 | Domain Expert (Bearing Diagnosis) | Major Revision | 5 |
| Reviewer 3 | Perspective Reviewer (ML Compression) | Minor Revision | 4 |
| Devil's Advocate | Critical Analyst | 3 CRITICAL issues flagged | 5 |

---

## Consensus Analysis

### Points of Agreement (Consensus)

**[CONSENSUS-5]** (All 5 reviewers agree):
1. The paper addresses a genuine gap — no prior study systematically compares pruning robustness across architectures for bearing fault diagnosis. The cross-architecture comparison is the paper's primary novel contribution.

2. File-level splitting (Section 3.3) is methodologically sound and a strength of the experimental design. All reviewers commend this choice.

3. The honest reporting of a falsified hypothesis (2D-CNN is not more pruning-robust) is a scientific strength. All reviewers note this positively.

4. The reproducibility statement and AI-use disclosure are comprehensive and meet best practices.

**[CONSENSUS-4]** (EIC, R1, R2, DA agree; R3 focuses on other aspects):
5. The FC architecture inclusion is problematic. EIC, R1, R2, and DA all note that FC at 691K params with near-chance accuracy is not a meaningful comparator and inflates the apparent scope.

**[CONSENSUS-4]** (EIC, R1, R3, DA agree; R2 notes but does not emphasize):
6. The ablation study's training budget confound (10-epoch fine-tune vs. 50-epoch scratch) is the single most important methodological weakness. EIC, R1, R3, and DA all identify this as a critical issue.

### Points of Disagreement

**Disagreement 1: Severity of the ablation confound**
- **R1 view**: Critical — undermines the paper's central claim (Methodology Report, W1)
- **R3 view**: Important but addressable with extended fine-tuning (Perspective Report, W1)
- **DA view**: CRITICAL — the confound alone could explain the observed gap (DA Report, DA-C1)
- **Disagreement type**: Severity disagreement
- **Editor's Resolution**: This is CRITICAL. The editor agrees with R1 and DA that the confound must be addressed before the paper's main conclusion is publishable. R3's more lenient view is noted but the conservative standard applies.
- **Resolution Rationale**: The ablation result is the paper's most novel contribution. If the experimental design cannot support the interpretation, the contribution is invalidated. Resolution: require matched-training-budget ablation.

**Disagreement 2: Overall assessment of practical contribution**
- **EIC view**: Significant gap but narrow scope limits contribution → Major Revision
- **R2 view**: Practical relevance is clear but limited by lab data → Major Revision
- **R3 view**: Modest but useful contribution, publishable with minor changes → Minor Revision
- **Disagreement type**: Severity disagreement
- **Editor's Resolution**: Major Revision is the appropriate decision. R3's perspective from the compression community is valid — the contribution is recognized as potentially valuable — but the methodological issues identified by EIC, R1, and DA must be resolved before publication.
- **Resolution Rationale**: The decision matrix (EIC=Major, R1=Major, R2=Major, R3=Minor) maps to Major Revision per editorial decision standards. The DA's CRITICAL issues also compel a Major Revision (cannot be Accept per IRON RULE).

---

## Devil's Advocate CRITICAL Issues — Editorial Response

Per Checkpoint Rule #4: *"If the Devil's Advocate finds CRITICAL issues, the Editorial Decision cannot be Accept."*

The DA identified three CRITICAL issues:

| DA-CRITICAL | Issue | Editorial Response |
|-------------|-------|-------------------|
| DA-C1 | Confounded ablation (10-epoch fine-tune vs. 50-epoch scratch) | **Upheld.** This must be addressed in revision. See Required Revision R1 below. |
| DA-C2 | Falsified hypothesis has multiple uncontrolled confounds (channel count, input quality, dataset suitability) | **Upheld in part.** The channel count confound is a genuine concern. The input quality and dataset suitability concerns are noted but less central. See Required Revision R4. |
| DA-C3 | Framing-deployment disconnect (zero-masking ≠ compression) | **Upheld.** The paper's framing must be revised to match the experimental methodology. See Required Revision R2. |

---

## Decision Rationale

This paper tackles a relevant and underexplored question — how different neural architectures fare under structured pruning for bearing fault diagnosis — and reports a useful negative result about L1 pruning's efficacy. The experimental design has several strengths (file-level splitting, multi-seed runs, reproducibility commitment), and the honest reporting of a falsified hypothesis is commendable.

However, the manuscript has three critical issues that must be addressed before publication. First, the central ablation comparison (prune-then-fine-tune vs. training from scratch) is confounded by unequal training budgets, which could fully explain the observed gap. Second, the paper's framing as a "compression" study is inconsistent with the use of zero-masking, which does not actually compress models. Third, the statistical characterization of results is insufficient — the absence of variance measures and formal hypothesis testing limits the reliability of all reported findings.

Four of five reviewers recommend Major Revision, and the Devil's Advocate's three CRITICAL issues compel a decision below Accept. The editorial decision is therefore **Major Revision** with the understanding that a substantially revised manuscript — one that addresses the ablation confound, reconciles the framing with the methodology, and provides appropriate statistical measures — will be sent for re-review.

---

## Required Revisions (Must Fix)

| # | Revision Item | Source Reviewer | Severity | Section | Estimated Effort |
|---|--------------|----------------|----------|---------|-----------------|
| R1 | Match training budgets in ablation study | EIC, R1, DA (DA-C1) | Critical | 5.4 | 2-3 days |
| R2 | Reconcile framing with zero-masking methodology | EIC, R3, DA (DA-C3) | Critical | 1, 5.3, 6.2 | 1 day |
| R3 | Report variance measures and add statistical testing | EIC, R1 | Critical | 5 | 2-3 days |
| R4 | Address channel count confound in architecture comparison | DA (DA-C2), R2 | Major | 5.1, 6.1 | 2-3 days |
| R5 | Expand experimental scope (random pruning baseline, additional pruning method, or both) | R1, R3 | Major | 4, 5 | 5-7 days |
| R6 | Replace or relegate FC architecture | EIC, R2, DA (DA-M1) | Major | 3.2, 5 | 1-2 days |

### Required Item Details

**R1: Match training budgets in ablation study**
- **Problem**: The ablation compares prune-then-fine-tune (10 epochs, LR 0.0005) vs. training from scratch (50 epochs, LR 0.001 with scheduling). The 5× training budget difference confounds the interpretation.
- **Source**: R1 (Methodology Report, W1); DA (DA-C1); EIC (W1)
- **Requirement**: Re-run the ablation with matched training budgets. Options: (a) extend fine-tuning to 50 epochs, (b) train scratch models for only 10 epochs, or (c) both. Report results for both budget-matched regimes. If the gap persists with matched budgets, the conclusion is strengthened. If it closes, the conclusion must be revised.
- **Acceptance criteria**: Ablation results must come from experiments where pruned and scratch models received the same number of training steps (or same total compute), with appropriate learning rate schedules for each condition.

**R2: Reconcile framing with zero-masking methodology**
- **Problem**: The paper frames itself around "model compression for edge deployment" but uses zero-masking that does not reduce model size or latency. Section 5.3 correctly identifies this limitation.
- **Source**: R3 (Perspective Report, W1); DA (DA-C3); EIC (W5)
- **Requirement**: Reframe the paper around "pruning robustness" (tolerance to parameter removal) rather than "compression." Revise the Abstract, Introduction, and Discussion to remove or qualify claims about compression and edge deployment that are not supported by zero-masking experiments. The latency analysis (Section 5.3) should be reframed as a cautionary result. Alternatively, add physical pruning experiments.
- **Acceptance criteria**: The paper's framing in title, abstract, and introduction must match the experimental methodology. Deployment-related claims must be limited to what zero-masking experiments can support.

**R3: Report variance measures and add statistical testing**
- **Problem**: All results (Tables 1, 2) report only mean accuracy across 3 seeds with no standard deviations or confidence intervals. The hypothesis falsification is asserted without statistical testing.
- **Source**: EIC (W2); R1 (W2, W3)
- **Requirement**: Report standard deviations or per-seed values for all experiments. Add at minimum a bootstrap confidence interval or effect size for the accuracy drop comparison between architectures. If resources permit, increase to 5-10 seeds.
- **Acceptance criteria**: All tables must include variance information. The hypothesis test (2D-CNN is not more pruning-robust) must be supported by a statistical comparison, not just visual inspection.

**R4: Address channel count confound in architecture comparison**
- **Problem**: 2D-CNN has 16 initial channels vs. 32 for 1D-CNN. Pruning 75% of 16 channels leaves only 4 active channels. This alone could explain the lower pruning robustness.
- **Source**: DA (DA-C2)
- **Requirement**: Add a controlled experiment where 2D-CNN channel counts are matched to 1D-CNN (e.g., 32-64-128 instead of 16-32-64). Alternatively, analyze per-layer pruning effects to show the contribution of initial channel count. If the channel count confound cannot be resolved, explicitly acknowledge it as a limitation.
- **Acceptance criteria**: The paper must either control for channel count or provide analysis showing that the finding is not driven by this confound.

**R5: Expand experimental scope with additional pruning controls**
- **Problem**: Only one pruning criterion (L1-norm magnitude) is tested. No random pruning baseline is included. This limits generalizability.
- **Source**: R1 (W5); R3 (W3); DA (DA-M4)
- **Requirement**: Add a random pruning baseline at minimum. Ideally, add one more pruning criterion (e.g., Taylor-first-order). This is essential for distinguishing architecture-specific pruning robustness from generic parameter redundancy effects.
- **Acceptance criteria**: At minimum, a random pruning control must be included in the main results. Results for an additional pruning criterion are strongly encouraged.

**R6: Replace or relegate FC architecture**
- **Problem**: FC at 691K params with near-chance accuracy is not a meaningful comparator. Its inclusion inflates the apparent scope and may mislead readers.
- **Source**: EIC (W4); R2 (W3); DA (DA-M1)
- **Requirement**: Either (a) replace FC with a competitive lightweight architecture of similar parameter count to the CNNs, or (b) move all FC results to supplementary material and explicitly caveat the floor effect in the main text. The Abstract and Conclusion must be updated accordingly.
- **Acceptance criteria**: The paper's primary comparison should be between architectures of comparable practical relevance. FC may appear in supplementary material but should not be presented as a primary result.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Source Reviewer | Priority | Section | Expected Improvement |
|---|--------------|----------------|----------|---------|---------------------|
| S1 | Connect ablation finding to general pruning literature (Liu et al. 2019, Renda et al. 2020, Frankle & Carbin 2019) | R3, DA | P2 | 5.4, 6.1 | Paper positioned in broader context |
| S2 | Add per-layer pruning sensitivity analysis | R3 | P2 | 5 | Explains *why* certain architectures suffer more |
| S3 | Expand noise robustness analysis (compare pruned vs. scratch at matched width) | R3, DA (DA-M5) | P2 | 5.2 | Strengthens practical relevance |
| S4 | Add STFT parameter details (window, hop, FFT size) | R1 (W4) | P3 | 3.3 | Enables full reproducibility |
| S5 | Expand training hyperparameter documentation (batch size, validation strategy, scheduler params) | R1 (W4) | P3 | 3.4 | Enables full reproducibility |
| S6 | Add a visualization of the accuracy-parameter Pareto frontier | R3, DA | P3 | 5 | Improves readability and practical utility |
| S7 | Add discussion of quantization and knowledge distillation as alternative compression methods | DA (Alt Path D) | P3 | 6.2 | Provides a complete picture for practitioners |

---

## Revision Roadmap

### Priority 1 — Structural Revisions (Estimated total effort: 2-3 weeks)
- [ ] R1: Re-run ablation with matched training budgets (extend fine-tuning to 50 epochs; optionally also train scratch for 10 epochs)
- [ ] R2: Reframe paper around "pruning robustness" rather than "compression"; revise Abstract, Introduction, Section 6.2
- [ ] R3: Add variance measures (SD or per-seed) to all tables; add statistical comparison for the hypothesis test
- [ ] R4: Control for channel count in architecture comparison or acknowledge as explicit limitation
- [ ] R5: Add random pruning baseline (minimum); add second pruning criterion (strongly encouraged)
- [ ] R6: Remove or relegate FC architecture from primary comparisons

### Priority 2 — Content Supplementation (Estimated total effort: 1-2 weeks)
- [ ] S1: Add connection to Liu et al. (2019), Renda et al. (2020), Frankle & Carbin (2019)
- [ ] S2: Add per-layer pruning sensitivity analysis
- [ ] S3: Expand noise robustness analysis to compare pruned vs. scratch models
- [ ] S5: Complete training hyperparameter documentation table

### Priority 3 — Text and Formatting (Estimated total effort: 3-5 days)
- [ ] S4: Add STFT parameter details
- [ ] S6: Add accuracy-parameter Pareto frontier visualization
- [ ] S7: Add brief discussion of alternative compression methods (quantization, distillation)
- [ ] Minor issues from all reviewers: verify MFPT FC ablation values, correct any typographical errors, fix Wang et al. (2025) reference year
- [ ] Embed Figures 1 and 2 in the manuscript (currently referenced but absent)

### Total Estimated Effort
- **Major Revision**: 4-6 weeks

---

## Revision Deadline

- **Recommended deadline**: 8 weeks from the date of this decision
- **Basis**: Major Revision standard (6-8 weeks, with the upper bound given the complexity of re-running ablation experiments)
- **Extension policy**: If extension is needed, please notify the editor 1 week before the deadline

---

## Response Letter Instructions

Please respond to every reviewer comment item by item using the format:
- **Reviewer comment**: [quote the specific comment]
- **Author response**: [your response]
- **Revision**: [description of what was changed and where]

**Must include**:
1. Response and revision description for each Required Revision (R1-R6)
2. Response for each Suggested Revision (S1-S7), indicating whether adopted and why, or providing a reason for not adopting
3. Cross-reference of changes (old/new text locations)

---

## Closing

We encourage you to carefully consider the reviewers' comments and submit a substantially revised manuscript. The reviewers have identified a promising core contribution — the first cross-architecture pruning robustness comparison in bearing fault diagnosis — but have also raised several critical methodological concerns that must be addressed. The strongest areas for improvement are: (1) matching training budgets in the ablation to support the central claim, (2) aligning the paper's framing with what the experiments actually test (pruning robustness, not compression), and (3) providing proper statistical characterization of results.

Please note that the revised manuscript will undergo another round of review, and the same reviewers will be asked to evaluate whether their concerns have been adequately addressed.

We look forward to receiving your revised manuscript.

Sincerely,
**Editorial Board**
*Mechanical Systems and Signal Processing* (simulated review)

---

## Appendix: Full Reviewer Reports

The five complete reviewer reports are included above in Phase 1 of this document.

---

*Review generated by academic-paper-reviewer skill v1.10.0 | Multi-Perspective Academic Paper Review Agent Team*
