
import os
import sys
import numpy as np
import pandas as pd
import librosa
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mir_eval.segment
import scipy.signal

sys.path.insert(0, str(Path(__file__).parent / "SpecTNT-pytorch"))
sys.path.insert(0, str(Path(__file__).parent))
from networks import SpecTNT, ResFrontEnd



AUDIO_DIR = r"F:\research module\salami_audio"
ANNOT_DIR = r"F:\research module\salami-data-public\annotations"
META_PATH = r"F:\research module\salami-data-public\metadata\metadata.csv"
CACHE_DIR = r"F:\research module\feature_cache"
RESULTS_CSV = r"F:\research module\results.csv"

SAMPLE_RATE = 16000
N_FFT       = 1024
HOP_LENGTH  = 512

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

TAXONOMY    = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence"]
LABEL_TO_ID = {l: i for i, l in enumerate(TAXONOMY)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}
N_CLASSES   = len(TAXONOMY)

PAPER = {"HR.5F":0.490, "ACC":0.544, "PWF":0.651, "Sf":0.632, "CHR.5F":0.357, "CF1":0.811}

os.makedirs(CACHE_DIR, exist_ok=True)

# STEP 1 — LABEL CONVERSION 


SUBSTRINGS = [
    ("silence","silence"),("pre-chorus","verse"),("prechorus","verse"),
    ("refrain","chorus"),("chorus","chorus"),("theme","chorus"),
    ("stutter","chorus"),("verse","verse"),("rap","verse"),
    ("section","verse"),("slow","verse"),("build","verse"),
    ("dialog","verse"),("intro","intro"),("fadein","intro"),
    ("opening","intro"),("bridge","bridge"),("trans","bridge"),
    ("out","outro"),("coda","outro"),("ending","outro"),
    ("break","inst"),("inst","inst"),("interlude","inst"),
    ("impro","inst"),("solo","inst"),
]

def convert_label(raw):
    if raw.strip().lower() == "end":
        return "end"
    for s1, s2 in SUBSTRINGS:
        if s1 in raw.lower():
            return s2
    return "inst"


# STEP 2 — LOAD ANNOTATIONS


def load_annotation(song_id):
    path = Path(ANNOT_DIR) / str(song_id) / "textfile1.txt"
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    entries.append((float(parts[0]), parts[1].strip()))
                except ValueError:
                    continue
    segs = []
    for i in range(len(entries) - 1):
        start, raw = entries[i]
        end = entries[i + 1][0]
        lab = convert_label(raw)
        if lab != "end" and end > start:
            segs.append({"start": start, "end": end, "label": lab})
    return segs

# STEP 3 — FEATURE EXTRACTION (all 6 features)

def extract_feature(audio, sr, name):
    if name == "harmonic":
        h = librosa.effects.harmonic(audio, margin=8)
        s = np.abs(librosa.stft(h, n_fft=N_FFT, hop_length=HOP_LENGTH))
        return librosa.amplitude_to_db(s, ref=np.max)
    elif name == "logmel":
        m = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT,
                                            hop_length=HOP_LENGTH, n_mels=128)
        return librosa.power_to_db(m, ref=np.max)
    elif name == "cqt":
        c = np.abs(librosa.cqt(audio, sr=sr, hop_length=HOP_LENGTH,
                                n_bins=84, bins_per_octave=12))
        return librosa.amplitude_to_db(c, ref=np.max)
    elif name == "chroma":
        return librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT,
                                            hop_length=HOP_LENGTH, n_chroma=12)
    elif name == "mfcc":
        mf = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
        d1 = librosa.feature.delta(mf)
        d2 = librosa.feature.delta(mf, order=2)
        return np.vstack([mf, d1, d2])
    elif name == "linear":
        s = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        return librosa.amplitude_to_db(s, ref=np.max)
    else:
        raise ValueError(f"Unknown feature: {name}")

FEATURE_DIMS = {
    "harmonic": N_FFT // 2 + 1,
    "logmel":   128,
    # "cqt":      84,
    # "chroma":   12,
    # "mfcc":     120,
    # "linear":   N_FFT // 2 + 1,
}


# STEP 4 — DATA AUGMENTATION 


def augment_audio(audio, sr):
    aug = audio.copy()
    if np.random.rand() < 0.5:
        aug = aug + np.random.uniform(0.001, 0.005) * np.random.randn(len(aug))
    if np.random.rand() < 0.5:
        aug = aug * np.random.uniform(0.7, 1.3)
    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            cutoff = np.random.uniform(2000, 6000)
            b, a = scipy.signal.butter(4, cutoff / (sr/2), btype="low")
        else:
            cutoff = np.random.uniform(100, 500)
            b, a = scipy.signal.butter(4, cutoff / (sr/2), btype="high")
        aug = scipy.signal.filtfilt(b, a, aug).astype(np.float32)
    if np.random.rand() < 0.2:
        steps = np.random.choice([-2, -1, 1, 2])
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=steps)
    if np.random.rand() < 0.2:
        rate = np.random.uniform(0.9, 1.1)
        st = librosa.effects.time_stretch(aug, rate=rate)
        aug = st[:len(aug)] if len(st) >= len(aug) else np.pad(st, (0, len(aug)-len(st)))
    return aug.astype(np.float32)


# STEP 5 — FEATURE CACHING

def get_cached_feature(song_id, feature_name):
    """Returns the clean (non-augmented) feature for a song, from cache if available."""
    cache_path = Path(CACHE_DIR) / f"{feature_name}_{song_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    # not cached — extract and save
    apath = Path(AUDIO_DIR) / f"{song_id}.wav"
    audio, sr = librosa.load(str(apath), sr=SAMPLE_RATE, mono=True)
    feat = extract_feature(audio, sr, feature_name)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    np.save(cache_path, feat)
    return feat


# STEP 6 — DATASET


class SalamiDataset(Dataset):
    def __init__(self, song_ids, feature_name, chunk_frames, time_pooling_factor,
                 augment=False, n_augment=1):
        self.chunks = []
        self.pool = time_pooling_factor
        self.eval_data = []
        frame_dur = HOP_LENGTH / SAMPLE_RATE

        for sid in tqdm(song_ids, desc=f"Building dataset [{feature_name}]"):
            apath = Path(AUDIO_DIR) / f"{sid}.wav"
            if not apath.exists():
                continue
            segs = load_annotation(sid)
            if not segs:
                continue

            # clean feature from cache
            try:
                feat = get_cached_feature(sid, feature_name)
            except Exception:
                continue

            n_frames = feat.shape[1]
            labels = np.full(n_frames, LABEL_TO_ID["silence"], dtype=np.int64)
            for seg in segs:
                sf = int(seg["start"] / frame_dur)
                ef = min(int(seg["end"] / frame_dur), n_frames)
                labels[sf:ef] = LABEL_TO_ID[seg["label"]]

            duration = n_frames * frame_dur

            # original chunks
            self._add_chunks(feat, labels, chunk_frames)
            # eval uses clean original only
            self.eval_data.append((feat, segs, duration))

            # augmented copies (training only) — re-extract from augmented audio
            if augment:
                audio, sr = librosa.load(str(apath), sr=SAMPLE_RATE, mono=True)
                for _ in range(n_augment):
                    aug_audio = augment_audio(audio, sr)
                    aug_feat  = extract_feature(aug_audio, sr, feature_name)
                    aug_feat  = (aug_feat - aug_feat.mean()) / (aug_feat.std() + 1e-6)
                    nf = aug_feat.shape[1]
                    if nf <= len(labels):
                        lab = labels[:nf]
                    else:
                        lab = np.pad(labels, (0, nf - len(labels)),
                                     constant_values=LABEL_TO_ID["silence"])
                    self._add_chunks(aug_feat, lab, chunk_frames)

    def _add_chunks(self, feat, labels, chunk_frames):
        n_frames = feat.shape[1]
        for start in range(0, n_frames - chunk_frames + 1, chunk_frames):
            fc = feat[:, start:start + chunk_frames]
            lc = labels[start:start + chunk_frames]
            lc_pooled = lc[::self.pool][:chunk_frames // self.pool]
            self.chunks.append((
                th.tensor(fc, dtype=th.float32),
                th.tensor(lc_pooled, dtype=th.long)
            ))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


# STEP 7 — BUILD MODEL


def build_model(feature_name, chunk_frames):
    F = FEATURE_DIMS[feature_name]
    fe_channels = 64
    freq_pooling = [2, 2, 2]
    time_pooling = [2, 2, 1]
    time_pool_factor = 2 * 2 * 1

    fe = ResFrontEnd(in_channels=1, out_channels=fe_channels,
                     freq_pooling=freq_pooling, time_pooling=time_pooling)

    n_freq_after = F // 8
    n_time_after = chunk_frames // time_pool_factor

    model = SpecTNT(
        fe_model=fe,
        n_channels=fe_channels,
        n_frequencies=n_freq_after,
        n_times=n_time_after,
        spectral_dmodel=64, spectral_nheads=4, spectral_dimff=128,
        temporal_dmodel=64, temporal_nheads=8, temporal_dimff=128,
        embed_dim=64, n_blocks=2, dropout=0.1,
        use_tct=False,
        n_classes=N_CLASSES
    )
    return model, time_pool_factor


# CTL LOSS


def ctl_loss(logits):
    probs = th.softmax(logits, dim=-1)
    diff  = probs[:, 1:, :] - probs[:, :-1, :]
    return (diff ** 2).mean()


# STEP 8 — EVALUATION METRICS


def frames_to_segments(frame_labels, frame_dur):
    if len(frame_labels) == 0:
        return []
    segs = []
    start = 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != frame_labels[i-1]:
            segs.append({"start": start*frame_dur, "end": i*frame_dur,
                         "label": ID_TO_LABEL[int(frame_labels[i-1])]})
            start = i
    segs.append({"start": start*frame_dur, "end": len(frame_labels)*frame_dur,
                 "label": ID_TO_LABEL[int(frame_labels[-1])]})
    return [s for s in segs if s["end"] - s["start"] > 0.01]

def to_mir_intervals(segments):
    bounds = np.array([s["start"] for s in segments] + [segments[-1]["end"]])
    ivs = np.column_stack([bounds[:-1], bounds[1:]])
    anns = [s["label"] for s in segments]
    keep = ivs[:,1] - ivs[:,0] > 0
    return ivs[keep], [a for a,k in zip(anns,keep) if k]

def compute_metrics(pred_segs, ref_segs):
    if not pred_segs or not ref_segs:
        return {k:0.0 for k in PAPER}
    pred_ivs, pred_ann = to_mir_intervals(pred_segs)
    ref_ivs,  ref_ann  = to_mir_intervals(ref_segs)
    if len(pred_ivs)==0 or len(ref_ivs)==0:
        return {k:0.0 for k in PAPER}

    _, _, hr5f = mir_eval.segment.detection(ref_ivs, pred_ivs, window=0.5, trim=True)

    fr = 10.0
    dur = max(ref_ivs[-1,1], pred_ivs[-1,1])
    n = int(dur*fr)+1
    def label_grid(segs):
        g = ["silence"]*n
        for s in segs:
            a, b = int(s["start"]*fr), min(int(s["end"]*fr), n)
            for j in range(a,b): g[j] = s["label"]
        return g
    pg, rg = label_grid(pred_segs), label_grid(ref_segs)
    acc = sum(p==r for p,r in zip(pg,rg))/n

    common_end = min(ref_ivs[-1, 1], pred_ivs[-1, 1])
    def trim_to(ivs, anns, end):
        out_ivs, out_anns = [], []
        for (s, e), a in zip(ivs, anns):
            if s >= end: break
            out_ivs.append([s, min(e, end)]); out_anns.append(a)
        return np.array(out_ivs), out_anns
    r_ivs, r_ann = trim_to(ref_ivs,  ref_ann,  common_end)
    p_ivs, p_ann = trim_to(pred_ivs, pred_ann, common_end)
    if len(r_ivs)==0 or len(p_ivs)==0:
        pwf = sf = 0.0
    else:
        r_ivs[0,0] = p_ivs[0,0] = 0.0
        r_ivs[-1,1] = p_ivs[-1,1] = common_end
        _, _, pwf = mir_eval.segment.pairwise(r_ivs, r_ann, p_ivs, p_ann)
        _, _, sf  = mir_eval.segment.nce(r_ivs, r_ann, p_ivs, p_ann)

    pc = [s for s in pred_segs if s["label"]=="chorus"]
    rc = [s for s in ref_segs  if s["label"]=="chorus"]
    if pc and rc:
        pcb = np.array([s["start"] for s in pc]+[pc[-1]["end"]])
        rcb = np.array([s["start"] for s in rc]+[rc[-1]["end"]])
        _, _, chr5f = mir_eval.segment.detection(
            np.column_stack([rcb[:-1],rcb[1:]]),
            np.column_stack([pcb[:-1],pcb[1:]]), window=0.5, trim=True)
    else:
        chr5f = 0.0

    pgb = ["chorus" if x=="chorus" else "other" for x in pg]
    rgb = ["chorus" if x=="chorus" else "other" for x in rg]
    def grid_to_iv(grid):
        ivs, anns, st = [], [], 0
        for i in range(1,len(grid)):
            if grid[i]!=grid[i-1]:
                ivs.append([st/fr, i/fr]); anns.append(grid[i-1]); st=i
        ivs.append([st/fr, len(grid)/fr]); anns.append(grid[-1])
        return np.array(ivs), anns
    piv, pan = grid_to_iv(pgb)
    riv, ran = grid_to_iv(rgb)
    riv[0,0] = piv[0,0] = 0.0
    common = min(riv[-1,1], piv[-1,1])
    riv[-1,1] = piv[-1,1] = common
    _, _, cf1 = mir_eval.segment.pairwise(riv, ran, piv, pan)

    return {"HR.5F":hr5f, "ACC":acc, "PWF":pwf, "Sf":sf, "CHR.5F":chr5f, "CF1":cf1}

def evaluate(model, eval_data, chunk_frames, pool):
    model.eval()
    frame_dur = (HOP_LENGTH / SAMPLE_RATE) * pool
    all_metrics = []
    with th.no_grad():
        for feat, ref_segs, duration in eval_data:
            n_frames = feat.shape[1]
            preds = []
            for start in range(0, n_frames, chunk_frames):
                fc = feat[:, start:start+chunk_frames]
                if fc.shape[1] < chunk_frames:
                    fc = np.pad(fc, ((0,0),(0,chunk_frames - fc.shape[1])))
                x = th.tensor(fc, dtype=th.float32).unsqueeze(0).to(DEVICE)
                logits = model(x)
                preds.append(logits.argmax(-1).squeeze(0).cpu().numpy())
            pred_frames = np.concatenate(preds)
            pred_segs = frames_to_segments(pred_frames, frame_dur)
            all_metrics.append(compute_metrics(pred_segs, ref_segs))
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in PAPER}


# RESULTS LOGGING


def save_results(feature_name, results):
    row = {"Feature": feature_name, **{k: round(results[k], 3) for k in PAPER}}
    if Path(RESULTS_CSV).exists():
        df = pd.read_csv(RESULTS_CSV)
        df = df[df["Feature"] != feature_name]   # remove old entry if re-running
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, index=False)
    print(f"  Results saved to {RESULTS_CSV}")

def print_summary():
    if not Path(RESULTS_CSV).exists():
        print("No results yet. Run some features first.")
        return
    df = pd.read_csv(RESULTS_CSV)
    metrics = ["HR.5F","ACC","PWF","Sf","CHR.5F","CF1"]
    print(f"\n{'='*72}")
    print(f"  FINAL COMPARISON — SALAMI-pop")
    print(f"{'='*72}")
    print(f"  {'Feature':<12}" + "".join(f"{m:>9}" for m in metrics))
    print(f"  {'-'*68}")
    print(f"  {'Paper':<12}" + "".join(f"{PAPER[m]:>9.3f}" for m in metrics))
    print(f"  {'-'*68}")
    for _, r in df.iterrows():
        print(f"  {r['Feature']:<12}" + "".join(f"{r[m]:>9.3f}" for m in metrics))
    print(f"{'='*72}")


# TRAIN + EVALUATE


def train(feature_name, song_ids, epochs, batch_size, chunk_seconds):
    print(f"\n{'='*60}")
    print(f"  TRAINING — feature: {feature_name.upper()}")
    print(f"  device: {DEVICE} | epochs: {epochs} | batch: {batch_size}")
    print(f"{'='*60}")

    frame_rate   = SAMPLE_RATE / HOP_LENGTH
    chunk_frames = int(chunk_seconds * frame_rate)
    chunk_frames = (chunk_frames // 4) * 4

    model, pool = build_model(feature_name, chunk_frames)
    model = model.to(DEVICE)

    ds = SalamiDataset(song_ids, feature_name, chunk_frames, pool,
                       augment=True, n_augment=1)
    if len(ds) == 0:
        print("  No chunks built. Check data paths.")
        return None
    print(f"  Dataset: {len(ds)} chunks (incl. augmented)")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = th.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    CTL_WEIGHT = 0.3

    model.train()
    for epoch in range(epochs):
        total_loss = total_ce = total_ctl = 0
        n_batches = 0
        for feats, labels in dl:
            feats  = feats.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(feats)
            T = min(logits.shape[1], labels.shape[1])
            logits_seq  = logits[:, :T, :]
            logits_flat = logits_seq.reshape(-1, N_CLASSES)
            labels_flat = labels[:, :T].reshape(-1)
            ce   = criterion(logits_flat, labels_flat)
            ctl  = ctl_loss(logits_seq)
            loss = ce + CTL_WEIGHT * ctl
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); total_ce += ce.item(); total_ctl += ctl.item()
            n_batches += 1
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={total_loss/n_batches:.4f}  "
                  f"(CE={total_ce/n_batches:.4f}  CTL={total_ctl/n_batches:.4f})")

    th.save(model.state_dict(), f"model_{feature_name}.pt")
    print(f"  Model saved to model_{feature_name}.pt")

    print(f"\n  Evaluating...")
    results = evaluate(model, ds.eval_data, chunk_frames, pool)
    print(f"\n  {'Metric':<10} {'Result':>8} {'Paper':>8}")
    print(f"  {'-'*28}")
    for k in PAPER:
        print(f"  {k:<10} {results[k]:>8.3f} {PAPER[k]:>8.3f}")

    save_results(feature_name, results)
    return results


# MAIN


def get_song_ids(max_songs=None):
    meta    = pd.read_csv(META_PATH)
    popular = meta[meta["CLASS"] == "popular"]["SONG_ID"].astype(str).tolist()
    valid   = []
    for sid in popular:
        if (Path(AUDIO_DIR) / f"{sid}.wav").exists() and \
           (Path(ANNOT_DIR) / sid / "textfile1.txt").exists():
            valid.append(sid)
    if max_songs:
        valid = valid[:max_songs]
    print(f"Found {len(valid)} usable songs")
    return valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--feature", type=str, default="harmonic",
                        help="harmonic/logmel/cqt/chroma/mfcc/linear")
    parser.add_argument("--summary", action="store_true",
                        help="Print final comparison table from results.csv")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        sys.exit(0)

    if args.mode == "test":
        songs      = get_song_ids(max_songs=5)
        epochs     = 5
        batch_size = 2
        chunk_sec  = 8
    else:
        songs      = get_song_ids()
        epochs     = 100
        batch_size = 4
        chunk_sec  = 24

    if not songs:
        print("No songs found. Check your paths.")
        sys.exit(1)

    train(args.feature, songs, epochs, batch_size, chunk_sec)



"""

WORKFLOW (one feature at a time):
  python train_spectnt.py --mode full --feature harmonic
  python train_spectnt.py --mode full --feature logmel
  python train_spectnt.py --mode full --feature cqt
  python train_spectnt.py --mode full --feature chroma
  python train_spectnt.py --mode full --feature mfcc
  python train_spectnt.py --mode full --feature linear

QUICK TEST (5 songs):
  python train_spectnt.py --mode test --feature harmonic

FINAL TABLE :
  python train_spectnt.py --summary

"""
