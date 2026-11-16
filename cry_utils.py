"""Kode bersama notebook baby_cry.ipynb dan ekspor ONNX untuk website."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
import torchaudio.transforms as T

SR, CLIP, MAX_LEN = 8000, 40000, 20 * 8000  # 8 kHz, potongan latih 5 dtk, input maks 20 dtk
CLASSES = ["hungry", "belly_pain", "burping", "cold_hot", "discomfort", "tired", "laugh", "noise", "silence"]
ID = {"hungry": "Lapar", "belly_pain": "Sakit perut", "burping": "Ingin sendawa", "cold_hot": "Kedinginan/kepanasan",
      "discomfort": "Tidak nyaman", "tired": "Lelah/mengantuk", "laugh": "Tertawa", "noise": "Bising (bukan tangisan)",
      "silence": "Hening"}
SUFFIX = {"hu": "hungry", "bp": "belly_pain", "bu": "burping", "ch": "cold_hot", "dc": "discomfort", "ti": "tired"}


def _source(folder, name, sr):
    """Nama file -> (label, grup, sumber) atau None bila file dibuang."""
    if m := re.match(r"^([0-9a-fA-F-]{36})-.*-(\w\w)\.wav$", name):  # donateacry: label ada di akhiran
        return SUFFIX[m[2]], "bayi_" + m[1].lower(), "donateacry"
    if m := re.match(r"^(\d+)c\.\w+$", name):
        if sr == 8000:  # salinan persis rekaman folder lain (lihat audit di notebook)
            return None
        return folder, f"{folder}_c{int(m[1]) // 4}", "c44k"
    if m := re.match(r"^(laugh_\d+|silence)\.\w+_(\d+)\.wav$", name):  # potongan satu rekaman panjang
        return folder, f"{m[1]}_{int(m[2]) // 4}", m[1].split("_")[0]
    if m := re.match(r"^(\d-\d+)-[A-Z]\.ogg$", name):  # ESC-50
        return folder, "esc_" + m[1], "esc50"
    raise ValueError(f"pola nama tidak dikenal: {folder}/{name}")


def inventory(root):
    """Daftar file bersih: path, label, grup (untuk split tanpa bocor), sumber, sr, durasi, md5 sinyal."""
    rows = []
    for p in sorted(Path(root).rglob("*.*")):
        folder = p.parent.name.replace(" ", "_")
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        src = _source(folder, p.name, sr)
        if src:
            rows.append(dict(path=str(p), label=src[0], group=src[1], source=src[2], sr=sr,
                             dur=len(x) / sr, md5=hashlib.md5(x.tobytes()).hexdigest()))
    return pd.DataFrame(rows).drop_duplicates("md5").reset_index(drop=True)


def load_wave(src):
    """Path / file-like -> mono 8 kHz float32, diulang sampai >= 5 dtk, dipotong maks 20 dtk."""
    if hasattr(src, "seek"):
        src.seek(0)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(1)
    if not x.size:
        raise ValueError("audio kosong")
    if sr != SR:
        x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
    x = x[:MAX_LEN]
    return np.tile(x, -(-CLIP // len(x)))[:max(CLIP, len(x))].astype(np.float32)


SPLITS = {"train": 0.65, "val": 0.15, "test": 0.20}


def split(df, seed=42):
    """Split berbasis grup (satu bayi / blok rekaman hanya di satu subset), seimbang per kelas.
    Greedy: grup kelas langka dan grup besar dulu; tiap grup masuk subset yang paling kurang
    dari targetnya, relatif terhadap ukuran kelas."""
    rng = np.random.default_rng(seed)
    cnt = pd.crosstab(df.group, df.label)  # baris = grup, kolom = label
    n = cnt.sum()
    order = cnt.index[rng.permutation(len(cnt))]
    key = pd.DataFrame({"rare": [n[cnt.loc[g] > 0].min() for g in order], "size": -cnt.loc[order].sum(1).values}, index=order)
    target = {s: f * n for s, f in SPLITS.items()}
    cur = {s: 0 * n for s in SPLITS}
    where = {}
    for g in key.sort_values(["rare", "size"], kind="stable").index:
        v = cnt.loc[g]
        where[g] = max(SPLITS, key=lambda s: ((target[s] - cur[s]) * v / n).sum())
        cur[where[g]] = cur[where[g]] + v
    return df.group.map(where).rename("split")


_MFCC = T.MFCC(SR, n_mfcc=20, melkwargs=dict(n_fft=512, hop_length=400, n_mels=40))


def mfcc(wave):
    """5 dtk tengah -> MFCC (101 frame x 20), dinormalisasi per koefisien (CMVN)."""
    s = max(0, (len(wave) - CLIP) // 2)
    m = _MFCC(torch.from_numpy(np.ascontiguousarray(wave[s:s + CLIP]))).T
    return ((m - m.mean(0)) / (m.std(0) + 1e-5)).numpy()


@torch.no_grad()
def dtw_dist(A, B, chunk=8):
    """Jarak DTW antar semua pasangan (A[i], B[j]); A (Na,T,D), B (Nb,T,D) panjang sama.
    D[i,j] = c[i,j] + min(D[i-1,j-1], D[i-1,j], D[i,j-1]), dihitung per anti-diagonal."""
    A, B = torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(B, dtype=torch.float32)
    t = A.shape[1]
    out = []
    for a in A.split(chunk):
        c = torch.cdist(a[:, None], B[None], compute_mode="donot_use_mm_for_euclid_dist")  # (na, Nb, T, T) jarak Euclid antar frame
        D = torch.full((*c.shape[:2], t + 1, t + 1), float("inf"))
        D[..., 0, 0] = 0
        for d in range(2, 2 * t + 1):
            i = torch.arange(max(1, d - t), min(t, d - 1) + 1)
            j = d - i
            best = torch.minimum(torch.minimum(D[..., i - 1, j - 1], D[..., i - 1, j]), D[..., i, j - 1])
            D[..., i, j] = c[..., i - 1, j - 1] + best
        out.append(D[..., t, t] / (2 * t))
    return torch.cat(out)


class CryCNN(nn.Module):
    """Sinyal mentah 8 kHz -> logit 9 kelas. Front-end (STFT, mel, log) ada di dalam model
    supaya file ONNX bisa langsung menerima sampel dari browser."""

    def __init__(self, n_fft=256, hop=80, n_mels=64, n_classes=len(CLASSES), p=0.3):
        super().__init__()
        n, k = torch.arange(n_fft), torch.arange(n_fft // 2 + 1)[:, None]
        win = torch.hann_window(n_fft)
        ang = 2 * torch.pi * k * n / n_fft
        self.register_buffer("dft", torch.cat([win * torch.cos(ang), -win * torch.sin(ang)])[:, None], persistent=False)
        self.register_buffer("fb", AF.melscale_fbanks(n_fft // 2 + 1, 0, SR / 2, n_mels, SR), persistent=False)
        self.hop, self.nb = hop, n_fft // 2 + 1
        self.augment = nn.Sequential(T.FrequencyMasking(8), T.TimeMasking(40))
        blocks, c = [nn.BatchNorm2d(1)], 1
        for c_out in (16, 32, 64, 128):
            blocks += [nn.Conv2d(c, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c = c_out
        self.cnn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(c, n_classes))

    def logmel(self, x):
        x = x / (x.abs().amax(1, keepdim=True) + 1e-8)  # normalisasi puncak: level mic tidak berpengaruh
        z = nn.functional.conv1d(x[:, None], self.dft, stride=self.hop)  # STFT sebagai konvolusi
        power = z[:, :self.nb] ** 2 + z[:, self.nb:] ** 2
        return (self.fb.T @ power + 1e-6).log()  # (B, mel, frame)

    def forward(self, x):
        s = self.logmel(x)
        s = (s - s.mean(-1, keepdim=True))[:, None]  # CMVN: buang warna mikrofon/ruangan per pita mel
        if self.training:
            s = self.augment(s)
        return self.head(self.cnn(s).mean((2, 3)))


if __name__ == "__main__":
    import io
    import numpy as np, soundfile as sf, torch, torchaudio
    df = inventory("data/Baby Crying Sounds")
    assert sorted(df.label.unique()) == sorted(CLASSES), df.label.unique()
    assert df.md5.duplicated().sum() == 0
    c8 = df[df.path.str.contains(r"[\\/]\d+c\.wav$") & (df.sr == 8000)]
    assert c8.empty, c8.head()
    s = split(df)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(df.group[s == a]) & set(df.group[s == b]), (a, b)
    assert all((s[df.label == c] == "test").any() for c in CLASSES)
    buf = io.BytesIO(); t = np.arange(44100) / 44100
    sf.write(buf, np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1), 44100, format="WAV")
    w = load_wave(buf)
    assert w.dtype == np.float32 and len(w) == CLIP, len(w)
    buf = io.BytesIO(); sf.write(buf, np.zeros(800), 8000, format="WAV")
    assert np.isfinite(load_wave(buf)).all()
    A = torch.from_numpy(np.stack([mfcc(w), mfcc(np.roll(w, 999) * .5)]))
    D = dtw_dist(A, A)
    assert D.shape == (2, 2) and torch.allclose(D, D.T, atol=1e-4) and D.diag().abs().max() < 1e-4
    m = CryCNN().eval()
    with torch.no_grad():
        assert m(torch.randn(2, CLIP)).shape == (2, 9) and m(torch.randn(1, 56000)).shape == (1, 9)
        x = torch.randn(1, CLIP); x = x / x.abs().max()
        ref = torchaudio.transforms.MelSpectrogram(SR, n_fft=256, hop_length=80, n_mels=64, f_max=4000, center=False)(x)
        assert torch.allclose(m.logmel(x), (ref + 1e-6).log(), atol=1e-3)
    print("ok")
