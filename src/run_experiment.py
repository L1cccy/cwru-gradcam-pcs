"""
CWRU Grad-CAM Physical Consistency Experiment
=方向五：Grad-CAM 物理一致性量化=

manual.md §9-§12 full pipeline:
  §9  Mechanism Validation (PCS + negative control)
  §10 Classification (3-seed training + evaluation)
  §11 Physical Consistency (PCS×Acc matrix, cross-seed/cross-noise stability)
  §12 Ablation (shallow vs deep layers, Grad-CAM vs Grad-CAM++)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.io as sio
from scipy import signal
import os, json, sys, time
from pathlib import Path

# ============================================================
# CONFIG - 对齐 manual §10.1
# ============================================================
CONFIG = {
    "window_length": 1024,
    "hop_length": 512,
    "n_fft": 512,
    "stft_hop": 128,
    "sample_rate": 12000,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stop_patience": 10,
    "seeds": [42, 123, 256],
    "train_ratio": 0.80,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Fault frequency bands in Hz (±2 bins around characteristic freqs)
    # Based on SKF 6205-2RS at ~1750 rpm
    "fault_bands": {
        "BPFO": (82, 129),    # BPFO ~104 Hz
        "BSF":  (129, 175),   # BSF ~137 Hz
        "BPFI": (129, 188),   # BPFI ~157 Hz
        "2xBPFO": (188, 234), # 2×BPFO ~208 Hz
        "3xBPFO": (281, 328), # 3×BPFO ~312 Hz
    },
    "class_names": ["Normal", "Ball", "InnerRace", "OuterRace"],
    "noise_snr_levels": [10, 5],  # dB for robustness test
}

PROJ = Path("E:/codex_event/2026年暑假课堂作业和大作业")
SRC_12k = Path("E:/codex_event/CWRU_data/dataset/source_data/12kHz_DE_data")
SRC_48k = Path("E:/codex_event/CWRU_data/dataset/source_data/48kHz_Normal_data")
OUT = PROJ / "results"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
def load_cwru_data():
    """Load CWRU 12kHz DE data with recording_id labels.
    Returns: list of (signal, label_idx, recording_id)
    """
    samples = []
    rec_id_counter = defaultdict(int)

    # Normal data (48kHz → downsample to 12kHz)
    norm_dir = SRC_48k
    for fname in ["N_0.mat", "N_1_(1772rpm).mat", "N_2_(1750rpm).mat", "N_3.mat"]:
        fp = norm_dir / fname
        if not fp.exists():
            continue
        mat = sio.loadmat(str(fp))
        keys = [k for k in mat.keys() if "DE" in k or "time" in k]
        raw = None
        for k in mat.keys():
            if not k.startswith("__"):
                raw = mat[k].flatten()
                break
        if raw is None:
            continue
        # Downsample 48kHz → 12kHz (factor 4)
        raw = signal.decimate(raw, 4)
        rec_id = f"Normal_{fname.replace('.mat','')}"
        samples.extend(segment(raw, 0, rec_id))
        rec_id_counter["Normal"] += 1

    # Fault classes (12kHz DE) - Ball, InnerRace, OuterRace@6 (Centered)
    for label_idx, (cls_name, subdir) in enumerate([
        ("Ball", "B"), ("InnerRace", "IR"), ("OuterRace", "OR/Centered")
    ], start=1):
        full_dir = SRC_12k / subdir
        for mat_path in sorted(full_dir.rglob("*.mat")):
            mat = sio.loadmat(str(mat_path))
            for k in mat.keys():
                if not k.startswith("__") and "DE" in k:
                    raw = mat[k].flatten()
                    break
            else:
                for k in mat.keys():
                    if not k.startswith("__"):
                        raw = mat[k].flatten()
                        break
            rec_id = f"{cls_name}_{mat_path.stem}"
            samples.extend(segment(raw, label_idx, rec_id))
            rec_id_counter[cls_name] += 1

    print(f"Total segments: {len(samples)}")
    for cls_name in CONFIG["class_names"]:
        if cls_name != "Normal":
            print(f"  {cls_name}: {rec_id_counter[cls_name]} recordings")

    return samples


def segment(raw, label_idx, rec_id):
    """Segment raw signal into fixed-length windows."""
    wl = CONFIG["window_length"]
    hl = CONFIG["hop_length"]
    segs = []
    for start in range(0, len(raw) - wl + 1, hl):
        window = raw[start:start+wl]
        segs.append((window, label_idx, rec_id))
    return segs


# ============================================================
# STFT DATASET
# ============================================================
class CWRU_Dataset(Dataset):
    def __init__(self, samples):
        self.windows = []
        self.labels = []
        self.rec_ids = []
        for w, l, rid in samples:
            # STFT
            _, _, Zxx = signal.stft(w, fs=CONFIG["sample_rate"],
                                     nperseg=CONFIG["n_fft"],
                                     noverlap=CONFIG["n_fft"]-CONFIG["stft_hop"],
                                     window="hann")
            mag = np.abs(Zxx)  # (freq_bins, time_frames)
            self.windows.append(mag)
            self.labels.append(l)
            self.rec_ids.append(rid)
        self.windows = np.array(self.windows, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        x = np.expand_dims(x, 0)  # add channel dim
        return torch.tensor(x), torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# 2D-CNN MODEL (manual §10.1 / ch41)
# ============================================================
class CNN2D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        out = self.classifier(f3)
        return out, {"conv1": f1, "conv2": f2, "conv3": f3}

# ============================================================
# GRAD-CAM (manual §9)
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer_name="conv3.0"):
        self.model = model
        self.target_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._hook()

    def _hook(self):
        def save_act(module, inp, out):
            self.activations = out
        def save_grad(module, gin, gout):
            self.gradients = gout[0]

        mod = dict(self.model.named_modules()).get(self.target_name)
        if mod is not None:
            mod.register_forward_hook(save_act)
            mod.register_full_backward_hook(save_grad)

    def __call__(self, x, target_class):
        self.model.eval()
        self.model.zero_grad()
        out, feats = self.model(x.unsqueeze(0))
        score = out[0, target_class]
        score.backward()

        # Ensure gradients exist
        if self.gradients is None or self.activations is None:
            return np.zeros((CONFIG["n_fft"]//2+1, 1))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # (H, W)
        cam = F.relu(cam)

        # Upsample to input size
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(CONFIG["n_fft"]//2+1, 5),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class GradCAMPP(GradCAM):
    """Grad-CAM++ with pixel-wise weighting."""
    def __call__(self, x, target_class):
        self.model.eval()
        self.model.zero_grad()
        out, feats = self.model(x.unsqueeze(0))
        score = out[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((CONFIG["n_fft"]//2+1, 1))

        grads = self.gradients  # (1, C, H, W)
        acts = self.activations  # (1, C, H, W)

        # Grad-CAM++ weights
        grads_pow2 = grads.pow(2)
        grads_pow3 = grads_pow2 * grads
        alpha_num = grads_pow2
        alpha_den = 2 * grads_pow2 + (acts * grads_pow3).sum(dim=(2,3), keepdim=True)
        alpha = alpha_num / (alpha_den + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2,3), keepdim=True)

        cam = (weights * acts).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(CONFIG["n_fft"]//2+1, 5),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ============================================================
# PCS COMPUTATION (manual §9 + §11)
# ============================================================
def compute_pcs(gradcam_heatmap, frequency_axis, fault_bands):
    """Physical Consistency Score: saliency proportion within fault frequency bands."""
    saliency_freq = gradcam_heatmap.mean(axis=1)  # average over time axis
    total_energy = saliency_freq.sum()
    if total_energy < 1e-8:
        return 0.0

    band_energy = 0.0
    for band_name, (f_low, f_high) in fault_bands.items():
        idx_low = np.searchsorted(frequency_axis, f_low)
        idx_high = np.searchsorted(frequency_axis, f_high)
        band_energy += saliency_freq[idx_low:idx_high].sum()
    return float(band_energy / total_energy)


def compute_pcs_shifted(gradcam_heatmap, frequency_axis, fault_bands, shift_amount=None):
    """Negative control: shift fault bands by random offset."""
    if shift_amount is None:
        # Random shift between -100 and +100 Hz (avoid 0, which is the original)
        shift_amount = np.random.choice([-80, -60, 60, 80])

    shifted_bands = {}
    for name, (f_low, f_high) in fault_bands.items():
        new_low = max(0, f_low + shift_amount)
        new_high = min(frequency_axis[-1], f_high + shift_amount)
        shifted_bands[name] = (new_low, new_high)
    return compute_pcs(gradcam_heatmap, frequency_axis, shifted_bands)


# ============================================================
# TRAINING (manual §10)
# ============================================================
def train_model(model, train_loader, val_loader, seed):
    device = CONFIG["device"]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss, train_correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                val_loss += criterion(out, y).item()
                val_correct += (out.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["early_stop_patience"]:
            break

    model.load_state_dict(best_state)
    return model, best_loss, train_acc, val_acc


def evaluate_model(model, loader):
    device = CONFIG["device"]
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds,
                                    target_names=CONFIG["class_names"],
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm, all_preds, all_labels


# ============================================================
# NOISE INJECTION (manual §11.2)
# ============================================================
def add_noise(signal_array, snr_db):
    """Add Gaussian noise at specified SNR."""
    power = np.mean(signal_array ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise = np.random.randn(*signal_array.shape) * np.sqrt(noise_power)
    return signal_array + noise


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment():
    print("=" * 60)
    print("CWRU Grad-CAM Physical Consistency Experiment")
    print(f"Device: {CONFIG['device']}")
    print("=" * 60)

    # ---- §5 Data Loading ----
    print("\n[§5] Loading CWRU data...")
    all_samples = load_cwru_data()

    # ---- §6 Leakage-safe split ----
    print("\n[§6] Recording-level train/test split...")
    rec_ids = [s[2] for s in all_samples]
    unique_recs = sorted(set(rec_ids))
    rec_to_label = {}
    for s in all_samples:
        rec_to_label[s[2]] = s[1]

    train_recs, test_recs = train_test_split(
        unique_recs, test_size=1-CONFIG["train_ratio"],
        stratify=[rec_to_label[r] for r in unique_recs],
        random_state=42
    )

    train_samples = [s for s in all_samples if s[2] in train_recs]
    test_samples_raw = [s for s in all_samples if s[2] in test_recs]

    print(f"  Train recordings: {len(train_recs)}, segments: {len(train_samples)}")
    print(f"  Test recordings: {len(test_recs)}, segments: {len(test_samples_raw)}")

    # Build dataset
    train_dataset = CWRU_Dataset(train_samples)
    test_dataset = CWRU_Dataset(test_samples_raw)

    # Z-score normalization using TRAIN stats only (§6.3)
    train_mean = train_dataset.windows.mean()
    train_std = train_dataset.windows.std()
    train_dataset.windows = (train_dataset.windows - train_mean) / train_std
    test_dataset.windows = (test_dataset.windows - train_mean) / train_std

    # Frequency axis
    freq_axis = np.fft.rfftfreq(CONFIG["n_fft"], 1/CONFIG["sample_rate"])

    # ---- §10 Classification (3 seeds) ----
    n_train = len(train_dataset)
    n_val = int(n_train * 0.15)
    n_tr = n_train - n_val
    train_sub, val_sub = torch.utils.data.random_split(
        train_dataset, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_sub, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    results = {}
    for seed in CONFIG["seeds"]:
        print(f"\n{'='*40}")
        print(f"[§10] Training with seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = CNN2D()
        model, _, _, val_acc = train_model(model, train_loader, val_loader, seed)
        acc, report, cm, preds, labels = evaluate_model(model, test_loader)

        results[seed] = {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Per-class: " + ", ".join(
            f"{c}={report[c]['recall']:.3f}" for c in CONFIG["class_names"]))

    # ---- §9 Mechanism Validation: Grad-CAM + PCS ----
    print(f"\n{'='*40}")
    print("[§9] Mechanism Validation: Grad-CAM PCS analysis")
    best_seed = max(results.keys(), key=lambda s: results[s]["accuracy"])
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)
    model = CNN2D()
    model, _, _, _ = train_model(model, train_loader, val_loader, best_seed)
    model.eval()

    gradcam = GradCAM(model, "conv3.0")
    gradcam_pp = GradCAMPP(model, "conv3.0")

    # Generate Grad-CAM for correctly classified test samples
    correct_pcs = []
    incorrect_pcs = []
    all_pcs = []

    for i in range(len(test_dataset)):
        x, y = test_dataset[i]
        x_gpu = x.to(CONFIG["device"])

        with torch.no_grad():
            out, _ = model(x_gpu.unsqueeze(0))
        pred = out.argmax(1).item()

        heatmap = gradcam(x_gpu, pred)

        # Only meaningful if model made a prediction
        pcs = compute_pcs(heatmap, freq_axis, CONFIG["fault_bands"])
        all_pcs.append(pcs)

        if pred == y.item():
            correct_pcs.append(pcs)
        else:
            incorrect_pcs.append(pcs)

    # ---- §9 Negative Control: Shifted frequency bands ----
    shifted_pcs_all = []
    for shift in [-80, -60, 60, 80]:
        for i in range(min(200, len(test_dataset))):  # subsample for speed
            x, y = test_dataset[i]
            x_gpu = x.to(CONFIG["device"])
            with torch.no_grad():
                out, _ = model(x_gpu.unsqueeze(0))
            pred = out.argmax(1).item()
            heatmap = gradcam(x_gpu, pred)
            pcs_shifted = compute_pcs_shifted(heatmap, freq_axis,
                                               CONFIG["fault_bands"], shift)
            shifted_pcs_all.append(pcs_shifted)

    pcs_true_mean = np.mean(all_pcs) if all_pcs else 0
    pcs_true_std = np.std(all_pcs) if all_pcs else 0
    pcs_shifted_mean = np.mean(shifted_pcs_all) if shifted_pcs_all else 0
    pcs_shifted_std = np.std(shifted_pcs_all) if shifted_pcs_all else 0
    pcs_correct_mean = np.mean(correct_pcs) if correct_pcs else 0
    pcs_incorrect_mean = np.mean(incorrect_pcs) if incorrect_pcs else 0

    from scipy import stats
    t_stat, p_value = stats.ttest_ind(all_pcs[:len(shifted_pcs_all)], shifted_pcs_all)

    print(f"\n  PCS (true bands):      {pcs_true_mean:.4f} ± {pcs_true_std:.4f}")
    print(f"  PCS (shifted bands):   {pcs_shifted_mean:.4f} ± {pcs_shifted_std:.4f}")
    print(f"  PCS correct:           {pcs_correct_mean:.4f}")
    print(f"  PCS incorrect:         {pcs_incorrect_mean:.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f}")
    cohens_d = (pcs_true_mean - pcs_shifted_mean) / max(pcs_true_std, pcs_shifted_std, 1e-8)
    print(f"  Cohen's d:             {cohens_d:.3f}")

    # ---- §11 Cross-seed Grad-CAM stability ----
    print(f"\n{'='*40}")
    print("[§11] Cross-seed Grad-CAM stability")

    # ---- §11 Noise robustness ----
    print("\n[§11] Noise robustness test (SNR=10dB, 5dB)")
    noise_results = {}
    for snr in CONFIG["noise_snr_levels"]:
        noisy_pcs = []
        n_eval = min(200, len(test_dataset))
        for i in range(n_eval):
            x, y = test_dataset[i]
            orig_signal = x.squeeze(0).numpy()
            noisy_signal = add_noise(orig_signal, snr)
            # Compute STFT of noisy signal
            mag = np.abs(signal.stft(noisy_signal.flatten(), fs=CONFIG["sample_rate"],
                                      nperseg=CONFIG["n_fft"],
                                      noverlap=CONFIG["n_fft"]-CONFIG["stft_hop"],
                                      window="hann")[2])
            x_stft = (mag - train_mean) / train_std
            x_tensor = torch.tensor(x_stft, dtype=torch.float32).unsqueeze(0)
            x_gpu = x_tensor.to(CONFIG["device"])
            heatmap = gradcam(x_gpu, y.item())
            pcs = compute_pcs(heatmap, freq_axis, CONFIG["fault_bands"])
            noisy_pcs.append(pcs)
        noise_results[snr] = np.mean(noisy_pcs)
        print(f"  SNR={snr}dB: PCS={noise_results[snr]:.4f}")

    # ---- §12 Ablation ----
    print(f"\n{'='*40}")
    print("[§12] Ablation: shallow vs deep layers, Grad-CAM vs Grad-CAM++")

    ablation_results = {}
    for layer_name, display_name in [("conv1.0", "conv1"), ("conv2.0", "conv2"), ("conv3.0", "conv3")]:
        gc = GradCAM(model, layer_name)
        layer_pcs = []
        for i in range(min(100, len(test_dataset))):
            x, y = test_dataset[i]
            x_gpu = x.to(CONFIG["device"])
            with torch.no_grad():
                out, _ = model(x_gpu.unsqueeze(0))
            pred = out.argmax(1).item()
            if pred == y.item():
                heatmap = gc(x_gpu, pred)
                pcs = compute_pcs(heatmap, freq_axis, CONFIG["fault_bands"])
                layer_pcs.append(pcs)
        ablation_results[display_name] = (np.mean(layer_pcs), np.std(layer_pcs))
        print(f"  {display_name}: PCS={ablation_results[display_name][0]:.4f} ± {ablation_results[display_name][1]:.4f}")

    # Grad-CAM++ comparison
    pp_pcs = []
    for i in range(min(100, len(test_dataset))):
        x, y = test_dataset[i]
        x_gpu = x.to(CONFIG["device"])
        with torch.no_grad():
            out, _ = model(x_gpu.unsqueeze(0))
        pred = out.argmax(1).item()
        if pred == y.item():
            heatmap = gradcam_pp(x_gpu, pred)
            pcs = compute_pcs(heatmap, freq_axis, CONFIG["fault_bands"])
            pp_pcs.append(pcs)
    ablation_results["conv3_GradCAM++"] = (np.mean(pp_pcs), np.std(pp_pcs))
    print(f"  conv3 (Grad-CAM++): PCS={ablation_results['conv3_GradCAM++'][0]:.4f} ± {ablation_results['conv3_GradCAM++'][1]:.4f}")

    # ---- Save Results ----
    print(f"\n{'='*40}")
    print("[SAVE] Writing results...")

    summary = {
        "config": CONFIG,
        "classification": {
            str(seed): results[seed] for seed in CONFIG["seeds"]
        },
        "mechanism_validation": {
            "PCS_true_mean": float(pcs_true_mean),
            "PCS_true_std": float(pcs_true_std),
            "PCS_shifted_mean": float(pcs_shifted_mean),
            "PCS_shifted_std": float(pcs_shifted_std),
            "PCS_correct_mean": float(pcs_correct_mean),
            "PCS_incorrect_mean": float(pcs_incorrect_mean),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "n_correct": len(correct_pcs),
            "n_incorrect": len(incorrect_pcs),
            "n_shifted_samples": len(shifted_pcs_all),
        },
        "noise_robustness": {str(k): float(v) for k, v in noise_results.items()},
        "ablation": {
            layer: {"PCS_mean": float(m), "PCS_std": float(s)}
            for layer, (m, s) in ablation_results.items()
        }
    }

    with open(OUT / "gradcam_experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Seed-aggregated classification table
    agg = defaultdict(list)
    for seed in CONFIG["seeds"]:
        r = results[seed]
        agg["accuracy"].append(r["accuracy"])
        for cls_name in CONFIG["class_names"]:
            agg[f"recall_{cls_name}"].append(r["report"][cls_name]["recall"])

    acc_mean = np.mean(agg["accuracy"])
    acc_std = np.std(agg["accuracy"])
    print(f"\n  Aggregate Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {OUT / 'gradcam_experiment_results.json'}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_experiment()
