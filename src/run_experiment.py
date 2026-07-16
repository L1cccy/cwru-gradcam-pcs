"""
CWRU + MFPT Grad-CAM PCS Experiment v2
Dual-dataset, 3-class, 5-seed, full figure generation
manual.md S9-S12
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.io as sio
from scipy import signal
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "window_len": 1024, "hop_len": 512, "n_fft": 512, "stft_hop": 128,
    "sample_rate": 12000, "batch_size": 32, "lr": 0.001,
    "epochs": 50, "patience": 10, "seeds": [42, 123, 256, 789, 1024],
    "train_ratio": 0.80, "device": "cuda" if torch.cuda.is_available() else "cpu",
    "class_names": ["Normal", "InnerRace", "OuterRace"],
    "fault_bands": {  # Hz, for typical bearing at ~1500-1800 rpm
        "BPFO": (82, 129), "BSF": (129, 175), "BPFI": (129, 188),
        "2xBPFO": (188, 234), "3xBPFO": (281, 328),
    },
    "noise_snr": [10, 5],
    "n_fft_bins": 257,  # n_fft/2+1
}

ROOT = Path("E:/codex_event/cwru-gradcam-pcs")
CWRU_12K = Path("E:/codex_event/CWRU_data/dataset/source_data/12kHz_DE_data")
CWRU_48K = Path("E:/codex_event/CWRU_data/dataset/source_data/48kHz_Normal_data")
MFPT_ROOT = Path("E:/codex_event/MFPT/data/MFPT Fault Data Sets")
FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
def load_cwru():
    samples = []
    # Normal (48kHz) → downsample 4x
    norm_map = {"N_0.mat": "N_0", "N_1_(1772rpm).mat": "N_1",
                "N_2_(1750rpm).mat": "N_2", "N_3.mat": "N_3"}
    for fname, rid in norm_map.items():
        fp = CWRU_48K / fname
        if not fp.exists(): continue
        mat = sio.loadmat(str(fp))
        raw = None
        for k in mat.keys():
            if not k.startswith("__"):
                raw = mat[k].flatten(); break
        if raw is None: continue
        raw = signal.decimate(raw, 4)
        segs = segment_signal(raw, 0, f"CWRU_{rid}")
        samples.extend(segs)

    # IR and OR (12kHz)
    for label, subdir in [(1, "IR"), (2, "OR/Centered")]:
        full = CWRU_12K / subdir
        for mp in sorted(full.rglob("*.mat")):
            mat = sio.loadmat(str(mp))
            raw = None
            for k in mat.keys():
                if not k.startswith("__") and "DE" in k:
                    raw = mat[k].flatten(); break
            if raw is None:
                for k in mat.keys():
                    if not k.startswith("__"):
                        raw = mat[k].flatten(); break
            if raw is None: continue
            rid = f"CWRU_{subdir.split('/')[-1]}_{mp.stem}"
            segs = segment_signal(raw, label, rid)
            samples.extend(segs)
    return samples


def load_mfpt():
    samples = []

    # Normal (baseline)
    base_dir = MFPT_ROOT / "1 - Three Baseline Conditions"
    for i in range(1, 4):
        mp = base_dir / f"baseline_{i}.mat"
        mat = sio.loadmat(str(mp))
        b = mat['bearing'][0, 0]
        gs = b['gs'].flatten()
        sr = int(b['sr'].flatten()[0])
        gs = resample_signal(gs, sr, 12000)
        segs = segment_signal(gs, 0, f"MFPT_N_{i}")
        samples.extend(segs)

    # Inner Race (vload)
    ir_dir = MFPT_ROOT / "4 - Seven Inner Race Fault Conditions"
    for i in range(1, 8):
        mp = ir_dir / f"InnerRaceFault_vload_{i}.mat"
        mat = sio.loadmat(str(mp))
        b = mat['bearing'][0, 0]
        gs = b['gs'].flatten()
        sr = int(b['sr'].flatten()[0])
        gs = resample_signal(gs, sr, 12000)
        segs = segment_signal(gs, 1, f"MFPT_IR_{i}")
        samples.extend(segs)

    # Outer Race (constant load)
    or_dir = MFPT_ROOT / "2 - Three Outer Race Fault Conditions"
    for i in range(1, 4):
        mp = or_dir / f"OuterRaceFault_{i}.mat"
        mat = sio.loadmat(str(mp))
        b = mat['bearing'][0, 0]
        gs = b['gs'].flatten()
        sr = int(b['sr'].flatten()[0])
        gs = resample_signal(gs, sr, 12000)
        segs = segment_signal(gs, 2, f"MFPT_OR_{i}")
        samples.extend(segs)

    # Outer Race (vload)
    ov_dir = MFPT_ROOT / "3 - Seven More Outer Race Fault Conditions"
    for i in range(1, 8):
        mp = ov_dir / f"OuterRaceFault_vload_{i}.mat"
        mat = sio.loadmat(str(mp))
        b = mat['bearing'][0, 0]
        gs = b['gs'].flatten()
        sr = int(b['sr'].flatten()[0])
        gs = resample_signal(gs, sr, 12000)
        segs = segment_signal(gs, 2, f"MFPT_OR_vload_{i}")
        samples.extend(segs)

    return samples


def resample_signal(sig, orig_sr, target_sr):
    """Resample signal to target sampling rate."""
    if orig_sr == target_sr:
        return sig
    from scipy.signal import resample_poly
    # Use integer ratio resampling
    ratio = target_sr / orig_sr
    n_out = int(len(sig) * ratio)
    return signal.resample(sig, n_out)


def segment_signal(raw, label, rec_id):
    wl, hl = CONFIG["window_len"], CONFIG["hop_len"]
    segs = []
    for start in range(0, len(raw) - wl + 1, hl):
        segs.append((raw[start:start+wl], label, rec_id))
    return segs


# ============================================================
# Dataset with STFT
# ============================================================
class BearDataset(Dataset):
    def __init__(self, samples, train_mean=None, train_std=None):
        self.mags = []
        self.labels = []
        self.rec_ids = []
        for w, l, rid in samples:
            _, _, Zxx = signal.stft(w, fs=CONFIG["sample_rate"],
                                     nperseg=CONFIG["n_fft"],
                                     noverlap=CONFIG["n_fft"]-CONFIG["stft_hop"],
                                     window="hann")
            self.mags.append(np.abs(Zxx).astype(np.float32))
            self.labels.append(l)
            self.rec_ids.append(rid)
        self.mags = np.array(self.mags)
        if train_mean is not None:
            self.mags = (self.mags - train_mean) / (train_std + 1e-8)
        self._mean, self._std = train_mean, train_std

    def __len__(self): return len(self.mags)
    def __getitem__(self, i):
        x = np.expand_dims(self.mags[i], 0)
        return torch.tensor(x), torch.tensor(self.labels[i], dtype=torch.long)

    def get_mean_std(self):
        return float(self.mags.mean()), float(self.mags.std())


# ============================================================
# MODEL
# ============================================================
class CNN2D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.clf = nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x):
        f1 = self.conv1(x); f2 = self.conv2(f1); f3 = self.conv3(f2)
        return self.clf(f3), {"conv1": f1, "conv2": f2, "conv3": f3}


# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, layer_name="conv3.0"):
        self.model = model
        self.grads, self.acts = None, None
        mod = dict(self.model.named_modules()).get(layer_name)
        if mod:
            mod.register_forward_hook(lambda m,i,o: setattr(self, '_a', o))
            mod.register_full_backward_hook(lambda m,gi,go: setattr(self, '_g', go[0]))
        self._a, self._g = None, None

    def __call__(self, x, target):
        self.model.eval(); self.model.zero_grad()
        out, _ = self.model(x.unsqueeze(0))
        out[0, target].backward()
        if self._a is None or self._g is None:
            return np.zeros((CONFIG["n_fft_bins"], 5))
        w = self._g.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * self._a).sum(dim=1).squeeze(0))
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(CONFIG["n_fft_bins"], 5), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def compute_pcs(heatmap, freq_axis, bands):
    sf = heatmap.mean(axis=1)
    total = sf.sum()
    if total < 1e-8: return 0.0
    band_sum = sum(sf[np.searchsorted(freq_axis, lo):np.searchsorted(freq_axis, hi)].sum() for lo, hi in bands.values())
    return float(band_sum / total)


def compute_pcs_shifted(heatmap, freq_axis, bands, shift):
    shifted = {}
    for n, (lo, hi) in bands.items():
        nl, nh = max(0, lo+shift), min(freq_axis[-1], hi+shift)
        shifted[n] = (nl, nh)
    return compute_pcs(heatmap, freq_axis, shifted)


# ============================================================
# TRAINING
# ============================================================
def train_one(model, train_dl, val_dl):
    dev = CONFIG["device"]; model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    crit = nn.CrossEntropyLoss()
    best_loss, best_sd, pat = float("inf"), None, 0
    for ep in range(CONFIG["epochs"]):
        model.train()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            out, _ = model(x); crit(out, y).backward(); opt.step()
        model.eval()
        vl_total = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                out, _ = model(x)
                vl_total += crit(out, y).item()
        vloss = vl_total / len(val_dl)
        if vloss < best_loss: best_loss = vloss; best_sd = {k:v.cpu().clone() for k,v in model.state_dict().items()}; pat = 0
        else: pat += 1
        if pat >= CONFIG["patience"]: break
    model.load_state_dict(best_sd); return model


def evaluate(model, dl):
    dev = CONFIG["device"]; model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(dev)
            out, _ = model(x)
            preds.extend(out.argmax(1).cpu().numpy()); labels.extend(y.numpy())
    p, l = np.array(preds), np.array(labels)
    acc = (p == l).mean()
    from sklearn.metrics import classification_report, confusion_matrix
    rpt = classification_report(l, p, target_names=CONFIG["class_names"], output_dict=True, zero_division=0)
    cm = confusion_matrix(l, p)
    return acc, rpt, cm


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_confusion_matrices(cm_list, accs, seeds, dataset_name, fname):
    n = len(cm_list); cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows * cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[None, :]
    for i, (cm, seed, acc) in enumerate(zip(cm_list, seeds, accs)):
        ax = axes[i//cols, i%cols]
        im = ax.imshow(cm, cmap='Blues')
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r,c]), ha='center', va='center', fontsize=8)
        ax.set_title(f'Seed {seed} (Acc={acc:.3f})')
        names = CONFIG["class_names"]
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    for j in range(i+1, rows*cols):
        axes[j//cols, j%cols].axis('off')
    plt.suptitle(f'Confusion Matrices — {dataset_name}', fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


def plot_pcs_box(pcs_true, pcs_shifted, dataset_name, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([pcs_true, pcs_shifted], tick_labels=['True Bands', 'Shifted Bands'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ECDC4'); bp['boxes'][1].set_facecolor('#FF6B6B')
    t, p = stats.ttest_ind(pcs_true, pcs_shifted)
    d = (np.mean(pcs_true)-np.mean(pcs_shifted))/max(np.std(pcs_true), np.std(pcs_shifted), 1e-8)
    ax.text(0.5, 0.95, f't={t:.2f}, p={p:.4f}\nCohen\'s d={d:.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=10)
    ax.set_ylabel('Physical Consistency Score (PCS)')
    ax.set_title(f'PCS: True vs Shifted Frequency Bands — {dataset_name}')
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


def plot_accuracy_bar(acc_dict, fname):
    datasets = list(acc_dict.keys())
    means = [np.mean(acc_dict[d]) for d in datasets]
    stds = [np.std(acc_dict[d]) for d in datasets]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(datasets, means, yerr=stds, color=['#4ECDC4', '#45B7D1'], capsize=8)
    for b, m in zip(bars, means): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{m:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.1)
    ax.set_title('Classification Accuracy (mean ± SD, 5 seeds)')
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


def plot_noise_curve(pcs_clean, pcs_noise, fname):
    snr_levels = [np.inf] + CONFIG["noise_snr"]
    pcs_vals = [pcs_clean] + pcs_noise
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(snr_levels, pcs_vals, 'o-', color='#45B7D1', linewidth=2, markersize=8)
    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Random expectation (~1%)')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('PCS')
    ax.set_title('Grad-CAM Physical Consistency vs Noise Level')
    ax.legend(); ax.invert_xaxis()
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


def plot_layer_pcs(layer_pcs, fname):
    names = list(layer_pcs.keys())
    means = [layer_pcs[n][0] for n in names]
    stds = [layer_pcs[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#FFE66D', '#4ECDC4', '#FF6B6B']
    ax.bar(names, means, yerr=stds, color=colors[:len(names)], capsize=6)
    ax.set_ylabel('PCS'); ax.set_xlabel('Conv Layer')
    ax.set_title('Layer-wise Physical Consistency Score')
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


def plot_pcs_acc_matrix(pcs_mean, acc_mean, dataset_name, fname):
    fig, ax = plt.subplots(figsize=(5, 4))
    x_pos = [0.3 if pcs_mean < 0.02 else 0.7 for _ in [dataset_name]]
    y_pos = [acc_mean for _ in [dataset_name]]
    colors = ['#FF6B6B' if y > 0.9 and pcs_mean < 0.02 else '#4ECDC4' for y in y_pos]
    ax.scatter(x_pos, y_pos, s=300, c=colors, edgecolors='black', zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Physical Consistency Score'); ax.set_ylabel('Accuracy')
    ax.set_title(f'PCS × Accuracy Matrix — {dataset_name}')
    # Quadrant labels
    ax.text(0.25, 0.97, 'Speculative\n(High Acc, Low PCS)', ha='center', fontsize=8, color='red')
    ax.text(0.75, 0.97, 'Ideal\n(High Acc, High PCS)', ha='center', fontsize=8, color='green')
    ax.text(0.25, 0.55, 'Poor', ha='center', fontsize=8, color='gray')
    ax.text(0.75, 0.55, 'Interpretable', ha='center', fontsize=8, color='gray')
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


def plot_example_stft(orig_segments, fname):
    """Plot example STFT spectrograms for each class."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, (cls_name, segs) in enumerate(orig_segments.items()):
        w = segs[0][0]
        f, t, Zxx = signal.stft(w, fs=CONFIG["sample_rate"],
                                 nperseg=CONFIG["n_fft"],
                                 noverlap=CONFIG["n_fft"]-CONFIG["stft_hop"],
                                 window="hann")
        im = axes[i].pcolormesh(t*1000, f, np.abs(Zxx), shading='gouraud', cmap='inferno')
        axes[i].set_title(cls_name); axes[i].set_xlabel('Time (ms)'); axes[i].set_ylabel('Freq (Hz)')
        axes[i].set_ylim(0, 1000)
        # Mark fault frequency bands
        for name, (flo, fhi) in CONFIG["fault_bands"].items():
            if name in ["BPFO", "BPFI"]:
                axes[i].axhline(y=flo, color='cyan', linestyle='--', alpha=0.5, linewidth=0.5)
    plt.colorbar(im, ax=axes, label='Magnitude')
    plt.suptitle('Example STFT Spectrograms — CWRU', fontweight='bold')
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=150); plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("Grad-CAM PCS Experiment v2: CWRU + MFPT")
    print(f"Device: {CONFIG['device']}, Seeds: {CONFIG['seeds']}")
    print("="*60)

    # ---- Load Data ----
    print("\nLoading CWRU...")
    cwru_samples = load_cwru()
    print(f"  CWRU: {len(cwru_samples)} segments")
    print("Loading MFPT...")
    mfpt_samples = load_mfpt()
    print(f"  MFPT: {len(mfpt_samples)} segments")

    # Collect per-class example signals for STFT visualization
    cwru_examples = defaultdict(list)
    for s in cwru_samples:
        cls = CONFIG["class_names"][s[1]]
        if len(cwru_examples[cls]) < 1:
            cwru_examples[cls].append(s)
    plot_example_stft(cwru_examples, "fig1_stft_examples.png")

    # ---- Split by recording_id ----
    all_results = {}
    all_results["cwru"] = run_pipeline(cwru_samples, "CWRU", "cwru")
    all_results["mfpt"] = run_pipeline(mfpt_samples, "MFPT", "mfpt")

    # ---- Cross-dataset generalization ----
    print("\n" + "="*40)
    print("Cross-dataset: CWRU train -> MFPT test")
    train_recs, _ = split_rec_ids(cwru_samples)
    cwru_train = [s for s in cwru_samples if s[2] in train_recs]
    cwru_train_ds = BearDataset(cwru_train)
    m, s = cwru_train_ds.get_mean_std()
    cwru_train_ds = BearDataset(cwru_train, m, s)
    mfpt_norm_ds = BearDataset(mfpt_samples, m, s)

    n_tr = int(len(cwru_train_ds) * 0.85)
    tr_ds, vl_ds = torch.utils.data.random_split(cwru_train_ds, [n_tr, len(cwru_train_ds)-n_tr],
                                                   generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_dl = DataLoader(mfpt_norm_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    xd_accs, xd_pcs = [], []
    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = CNN2D()
        model = train_one(model, tr_dl, vl_dl)
        acc, rpt, cm = evaluate(model, test_dl)
        xd_accs.append(acc)
        print(f"  Seed {seed}: CWRU→MFPT Acc={acc:.4f}")
    all_results["cross_dataset"] = {"accs": xd_accs, "acc_mean": float(np.mean(xd_accs)), "acc_std": float(np.std(xd_accs))}

    # ---- Save ----
    with open(ROOT / "results" / "experiment_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Figures: {FIG_DIR}")
    print(f"Results: {ROOT / 'results' / 'experiment_results_v2.json'}")
    print("="*60)


def split_rec_ids(samples):
    recs = sorted(set(s[2] for s in samples))
    r2l = {}
    for s in samples: r2l[s[2]] = s[1]
    train, test = train_test_split(recs, test_size=1-CONFIG["train_ratio"],
                                    stratify=[r2l[r] for r in recs], random_state=42)
    return train, test


def run_pipeline(samples, name, prefix):
    print(f"\n{'='*40}")
    print(f"[{name}] Full pipeline")
    train_recs, test_recs = split_rec_ids(samples)
    train_s = [s for s in samples if s[2] in train_recs]
    test_s = [s for s in samples if s[2] in test_recs]
    print(f"  Train: {len(train_recs)} recs, {len(train_s)} segs")
    print(f"  Test:  {len(test_recs)} recs, {len(test_s)} segs")

    train_ds = BearDataset(train_s)
    m, s_std = train_ds.get_mean_std()
    train_ds = BearDataset(train_s, m, s_std)
    test_ds = BearDataset(test_s, m, s_std)

    n_tr = int(len(train_ds) * 0.85)
    tr_ds, vl_ds = torch.utils.data.random_split(train_ds, [n_tr, len(train_ds)-n_tr],
                                                   generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    freq_axis = np.fft.rfftfreq(CONFIG["n_fft"], 1/CONFIG["sample_rate"])

    # ---- S10: Classification ----
    accs, cms, best_model = [], [], None
    best_acc = 0
    for seed in CONFIG["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = CNN2D()
        model = train_one(model, tr_dl, vl_dl)
        acc, rpt, cm = evaluate(model, test_dl)
        accs.append(acc); cms.append(cm)
        print(f"  Seed {seed}: Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc; best_model = model
    plot_confusion_matrices(cms, accs, CONFIG["seeds"][:3], name, f"{prefix}_confusion.png")

    acc_mean = float(np.mean(accs)); acc_std = float(np.std(accs))

    # ---- S9+S11: Grad-CAM PCS + Negative Control ----
    best_model.eval(); gc = GradCAM(best_model, "conv3.0")
    pcs_true, pcs_shifted = [], []
    n_test = min(300, len(test_ds))
    shifts = [-80, -60, 60, 80]
    for i in range(n_test):
        x, yt = test_ds[i]; xd = x.to(CONFIG["device"])
        with torch.no_grad():
            out, _ = best_model(xd.unsqueeze(0))
        pred = out.argmax(1).item()
        hm = gc(xd, pred)
        pcs = compute_pcs(hm, freq_axis, CONFIG["fault_bands"])
        pcs_true.append(pcs)
        # Negative control (1 shift per sample)
        sh = shifts[i % len(shifts)]
        pcs_shifted.append(compute_pcs_shifted(hm, freq_axis, CONFIG["fault_bands"], sh))
    plot_pcs_box(pcs_true, pcs_shifted, name, f"{prefix}_pcs_box.png")

    t_stat, p_val = stats.ttest_ind(pcs_true, pcs_shifted)
    pc_true = float(np.mean(pcs_true)); pc_shift = float(np.mean(pcs_shifted))
    cd = (pc_true - pc_shift) / max(float(np.std(pcs_true)), float(np.std(pcs_shifted)), 1e-8)
    print(f"  PCS_true={pc_true:.4f}, PCS_shifted={pc_shift:.4f}, t={t_stat:.2f}, p={p_val:.4f}, d={cd:.2f}")

    # PCS × Acc matrix
    plot_pcs_acc_matrix(pc_true, acc_mean, name, f"{prefix}_pcs_acc_matrix.png")

    # ---- S11: Noise robustness (add noise to STFT magnitude directly) ----
    noise_pcs = []
    for snr in CONFIG["noise_snr"]:
        npcs = []
        for i in range(min(100, len(test_ds))):
            x, yt = test_ds[i]
            mag = x.squeeze(0).numpy()
            noise_rms = np.sqrt(np.mean(mag**2)) / (10**(snr/20))
            noisy_mag = mag + np.random.randn(*mag.shape).astype(np.float32) * noise_rms
            x_norm = (noisy_mag - m) / (s_std + 1e-8)
            xt = torch.tensor(x_norm).unsqueeze(0).to(CONFIG["device"])
            hm = gc(xt, yt.item())
            npcs.append(compute_pcs(hm, freq_axis, CONFIG["fault_bands"]))
        noise_pcs.append(float(np.mean(npcs)))
        print(f"  SNR={snr}dB: PCS={noise_pcs[-1]:.4f}")
    plot_noise_curve(pc_true, noise_pcs, f"{prefix}_noise_curve.png")

    # ---- S12: Ablation ----
    layer_pcs = {}
    for lname, dname in [("conv1.0", "conv1"), ("conv2.0", "conv2"), ("conv3.0", "conv3")]:
        gc_l = GradCAM(best_model, lname)
        lpcs = []
        for i in range(min(100, len(test_ds))):
            x, yt = test_ds[i]; xd = x.to(CONFIG["device"])
            with torch.no_grad(): out, _ = best_model(xd.unsqueeze(0))
            pred = out.argmax(1).item()
            if pred == yt.item():
                lpcs.append(compute_pcs(gc_l(xd, pred), freq_axis, CONFIG["fault_bands"]))
        layer_pcs[dname] = (float(np.mean(lpcs)), float(np.std(lpcs)))
        print(f"  {dname}: PCS={layer_pcs[dname][0]:.4f}±{layer_pcs[dname][1]:.4f}")
    plot_layer_pcs(layer_pcs, f"{prefix}_layer_pcs.png")

    return {
        "accuracies": [float(a) for a in accs],
        "acc_mean": acc_mean, "acc_std": acc_std,
        "pcs_true": pc_true, "pcs_shifted": pc_shift,
        "pcs_t_stat": float(t_stat), "pcs_p_value": float(p_val),
        "cohens_d": float(cd),
        "noise_pcs": noise_pcs,
        "layer_pcs": layer_pcs,
        "cross_dataset": {},
    }


if __name__ == "__main__":
    main()
