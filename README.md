# Grad-CAM Physical Consistency Score (PCS) for Bearing Fault Diagnosis

**Repository**: https://github.com/L1cccy/cwru-gradcam-pcs

## Research Question

In correctly classified CWRU and MFPT bearing fault samples, what proportion of Grad-CAM saliency concentrates within theoretical fault characteristic frequency bands? Does this physical consistency remain stable across random seeds, operating conditions, noise levels, and datasets?

## Datasets

- **CWRU** (12 kHz Drive End): 48 recordings, 3 classes (Normal, InnerRace, OuterRace)
- **MFPT** (resampled to 12 kHz): Normal + InnerRace + OuterRace fault conditions

## Method

STFT spectrogram (n_fft=512) + 2D-CNN with Grad-CAM heatmap analysis. Physical Consistency Score (PCS) = proportion of Grad-CAM saliency energy within theoretical fault frequency bands (BPFO/BPFI/BSF). Negative control via frequency-band shifting.

## Key Finding

Despite 92.7% classification accuracy, Grad-CAM attention concentrates only 0.9% within theoretical fault frequency bands — significantly less than random chance (2.9%). This suggests the model uses frequency patterns unrelated to bearing fault physics, qualifying as a "speculative model" (sensu ch49).

## Reproducibility

```bash
conda create -n cwru python=3.12
conda activate cwru
pip install torch scipy scikit-learn matplotlib
python src/run_experiment.py
```

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA)

## AI-Use Disclosure

AI assistance was used for: literature search, code generation, figure generation, manuscript drafting, and simulated peer review. All claims were verified against experimental evidence. No results or citations were fabricated.
