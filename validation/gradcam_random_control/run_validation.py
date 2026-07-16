"""
Validation: Grad-CAM Random Weight Baseline
Compare PCS between trained model and randomly initialized model.
If both produce similar PCS → Grad-CAM implementation is broken (outputs noise).
If PCS differs significantly → Grad-CAM is producing meaningful attention maps.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.io as sio, scipy.signal as signal
from pathlib import Path
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path("E:/codex_event/cwru-gradcam-pcs")
OUT = BASE / "validation" / "gradcam_random_control"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ========== Shared Config ==========
SAMPLE_RATE = 12000
N_FFT = 512
HOP_FFT = 128
WINDOW_LEN = 1024
HOP_LEN = 512
CLASS_NAMES = ["Normal", "InnerRace", "OuterRace"]
FAULT_BANDS = {  # same as main experiment
    "BPFO": (82, 129), "BSF": (129, 175), "BPFI": (129, 188),
    "2xBPFO": (188, 234), "3xBPFO": (281, 328),
}

# ========== Model (same as main experiment) ==========
class CNN2D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.clf = nn.Sequential(nn.Flatten(), nn.Linear(64,128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_classes))
    def forward(self, x):
        f1=self.conv1(x); f2=self.conv2(f1); f3=self.conv3(f2)
        return self.clf(f3), {"conv1":f1, "conv2":f2, "conv3":f3}

# ========== Grad-CAM ==========
class GradCAM:
    def __init__(self, model, layer="conv3.0"):
        self.model = model
        mod = dict(model.named_modules()).get(layer)
        mod.register_forward_hook(lambda m,i,o: setattr(self,'a',o))
        mod.register_full_backward_hook(lambda m,gi,go: setattr(self,'g',go[0]))
    def __call__(self, x, target):
        self.model.eval(); self.model.zero_grad()
        out,_ = self.model(x.unsqueeze(0))
        out[0,target].backward()
        if self.a is None or self.g is None: return np.zeros((257,5))
        w=self.g.mean(dim=(2,3),keepdim=True)
        cam=F.relu((w*self.a).sum(dim=1).squeeze(0))
        cam=cam.unsqueeze(0).unsqueeze(0)
        cam=F.interpolate(cam,size=(257,5),mode="bilinear",align_corners=False)
        cam=cam.squeeze().detach().cpu().numpy()
        return (cam-cam.min())/(cam.max()-cam.min()+1e-8)

# ========== Data Loading ==========
def load_all():
    samples = []
    c12k = Path("E:/codex_event/CWRU_data/dataset/source_data/12kHz_DE_data")
    c48k = Path("E:/codex_event/CWRU_data/dataset/source_data/48kHz_Normal_data")
    # Normal
    for fn, rid in [("N_0.mat","N_0"),("N_1_(1772rpm).mat","N_1"),("N_2_(1750rpm).mat","N_2"),("N_3.mat","N_3")]:
        raw = sio.loadmat(str(c48k/fn))
        for k in raw:
            if k.startswith("__"): continue
            v = signal.decimate(raw[k].flatten(), 4)
            break
        for s in range(0, len(v)-WINDOW_LEN+1, HOP_LEN):
            samples.append((v[s:s+WINDOW_LEN], 0, rid))
    # IR, OR
    for label, sub in [(1,"IR"), (2,"OR/Centered")]:
        for fp in sorted((c12k/sub).rglob("*.mat")):
            raw = sio.loadmat(str(fp))
            for k in raw:
                if not k.startswith("__"):
                    v = raw[k].flatten(); break
            rid = f"{sub.split('/')[-1]}_{fp.stem}"
            for s in range(0, len(v)-WINDOW_LEN+1, HOP_LEN):
                samples.append((v[s:s+WINDOW_LEN], label, rid))
    return samples

print("Loading data...")
samples = load_all()
rec_ids = sorted(set(s[2] for s in samples))
r2l = {s[2]: s[1] for s in samples}
train_recs, test_recs = train_test_split(rec_ids, test_size=0.2, stratify=[r2l[r] for r in rec_ids], random_state=42)
train_s = [s for s in samples if s[2] in train_recs]
test_s  = [s for s in samples if s[2] in test_recs]

def to_stft(win):
    _,_,Z=signal.stft(win, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_FFT, window="hann")
    return np.abs(Z).astype(np.float32)

train_mags = np.array([to_stft(s[0]) for s in train_s])
train_mean, train_std = train_mags.mean(), train_mags.std()
test_mags = np.array([to_stft(s[0]) for s in test_s])
train_mags = (train_mags - train_mean) / train_std
test_mags  = (test_mags - train_mean) / train_std

test_labels = np.array([s[1] for s in test_s])
test_ids = [s[2] for s in test_s]
freq_axis = np.fft.rfftfreq(N_FFT, 1/SAMPLE_RATE)

def compute_pcs(hm, bands):
    sf = hm.mean(axis=1)
    t = sf.sum()
    if t < 1e-8: return 0.0
    be = sum(sf[np.searchsorted(freq_axis,lo):np.searchsorted(freq_axis,hi)].sum() for lo,hi in bands.values())
    return be/t

def compute_pcs_shifted(hm, bands, shift):
    sb = {}
    for n,(l,h) in bands.items():
        nl,nh = max(0,l+shift), min(freq_axis[-1],h+shift)
        sb[n] = (nl,nh)
    return compute_pcs(hm, sb)

# ========== Random Model ==========
print("\n[TRAINED MODEL PCS - loading from experiment results]")
with open(BASE / "results" / "experiment_results_v2.json") as f:
    exp = json.load(f)
cwru_pcs_true_trained = exp["cwru"]["pcs_true"]
cwru_pcs_shift_trained = exp["cwru"]["pcs_shifted"]
print(f"  Trained model PCS_true:   {cwru_pcs_true_trained:.4f}")
print(f"  Trained model PCS_shifted: {cwru_pcs_shift_trained:.4f}")

print("\n[RANDOM MODEL - no training, random weights]")
torch.manual_seed(42)
np.random.seed(42)
model = CNN2D().to(DEVICE)
# Do NOT train - weights remain as initialized
model.eval()
gc = GradCAM(model, "conv3.0")

pcs_true_rand, pcs_shift_rand = [], []
n_test = min(300, len(test_mags))
for i in range(n_test):
    x = torch.tensor(test_mags[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out,_ = model(x.unsqueeze(0))
    pred = out.argmax(1).item() if out.max().item() > 0 else 0
    pred = out.argmax(1).item()
    hm = gc(x, pred)
    pcs_true_rand.append(compute_pcs(hm, FAULT_BANDS))
    shift = [-80,-60,60,80][i%4]
    pcs_shift_rand.append(compute_pcs_shifted(hm, FAULT_BANDS, shift))

rt = np.mean(pcs_true_rand)
rs = np.mean(pcs_shift_rand)
print(f"  Random model PCS_true:   {rt:.4f}")
print(f"  Random model PCS_shifted: {rs:.4f}")

# ========== Compare ==========
print("\n========== COMPARISON ==========")
print(f"{'':20s} {'Trained':>10s} {'Random':>10s} {'Random Expected':>15s}")
print(f"{'PCS_true':20s} {cwru_pcs_true_trained:>10.4f} {rt:>10.4f} {'~0.033 (chance)':>15s}")
print(f"{'PCS_shifted':20s} {cwru_pcs_shift_trained:>10.4f} {rs:>10.4f} {'~0.033 (chance)':>15s}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(pcs_true_rand, bins=30, alpha=0.7, label=f"Random (mean={rt:.3f})")
axes[0].axvline(cwru_pcs_true_trained, color='r', ls='--', lw=2, label=f"Trained ({cwru_pcs_true_trained:.3f})")
axes[0].set_xlabel("PCS"); axes[0].set_ylabel("Count"); axes[0].set_title("PCS_true comparison")
axes[0].legend()

axes[1].bar(["Trained PCS_true", "Trained PCS_shift", "Random PCS_true", "Random PCS_shift"],
            [cwru_pcs_true_trained, cwru_pcs_shift_trained, rt, rs])
axes[1].set_ylabel("PCS"); axes[1].tick_params(labelsize=8)
plt.suptitle("Grad-CAM Random Weight Baseline Validation")
plt.tight_layout()
plt.savefig(OUT / "random_vs_trained_pcs.png", dpi=150)

# Save results
results = {
    "trained_pcs_true": cwru_pcs_true_trained,
    "trained_pcs_shifted": cwru_pcs_shift_trained,
    "random_pcs_true": float(rt),
    "random_pcs_shifted": float(rs),
    "random_pcs_true_all": [float(v) for v in pcs_true_rand],
    "n_samples": n_test,
    "note": "Random model weights are untrained (default init). If PCS values are similar to trained model, Grad-CAM implementation is likely incorrect."
}
with open(OUT / "random_control_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {OUT}")
