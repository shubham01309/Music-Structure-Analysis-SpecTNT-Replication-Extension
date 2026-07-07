"""
USAGE:
  python train_harmonic_cnn.py --mode test
  python train_harmonic_cnn.py --mode full
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import argparse
from pathlib import Path
from tqdm import tqdm

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mir_eval.segment
import scipy.signal


# CONFIGURATION 


AUDIO_DIR   = r"F:\research module\salami_audio"
ANNOT_DIR   = r"F:\research module\salami-data-public\annotations"
META_PATH   = r"F:\research module\salami-data-public\metadata\metadata.csv"
CACHE_DIR   = r"F:\research module\feature_cache"
RESULTS_CSV = r"F:\research module\results_harmonic_cnn.csv"

SAMPLE_RATE   = 16000   # audia is resampled ti 16kHz
N_FFT         = 1024    #STFT window size
HOP_LENGTH    = 512     # STFT hope size
CHUNK_SECONDS = 24
EPOCHS        = 100
BATCH_SIZE    = 4
LEARNING_RATE = 0.0005
PATIENCE      = 2

# Loss weights 
BOUNDARY_WEIGHT = 0.9
FUNCTION_WEIGHT = 0.1

DEVICE = "cuda" if th.cuda.is_available() else "cpu"

TAXONOMY    = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence"]
LABEL_TO_ID = {l: i for i, l in enumerate(TAXONOMY)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}
N_CLASSES   = len(TAXONOMY)

# Harmonic-CNN results on Harmonix for comparison
PAPER = {
    "HR.5F":  0.559,
    "ACC":    0.680,
    "PWF":    0.670,
    "Sf":     0.682,
    "CHR.5F": 0.462,
    "CF1":    0.784
}

os.makedirs(CACHE_DIR, exist_ok=True)


# STEP 1 — LABEL CONVERSION (Algorithm 1)


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
    if raw.strip().lower() == "end": # converting raw label contains left string converting into right string
        return "end"
    for s1, s2 in SUBSTRINGS:
        if s1 in raw.lower():
            return s2
    return "inst"


# STEP 2 — LOAD ANNOTATIONS


def load_annotation(song_id):
    path = Path(ANNOT_DIR) / str(song_id) / "textfile1.txt" #path
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:              # if file doesnt exist
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")  #salami files are tab sapareted
            if len(parts) >= 2:
                try:
                    entries.append((float(parts[0]), parts[1].strip())) # second colunm raw label
                except ValueError:
                    continue
    segs = []
    for i in range(len(entries) - 1):
        start, raw = entries[i] 
        end = entries[i + 1][0]
        lab = convert_label(raw) # start at one timestamp and end at next timestemp
        if lab != "end" and end > start:
            segs.append({"start": start, "end": end, "label": lab})
    return segs


# STEP 3 — HARMONIC REPRESENTATION


def extract_harmonic(audio, sr):
    h = librosa.effects.harmonic(audio, margin=8)
    s = np.abs(librosa.stft(h, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return librosa.amplitude_to_db(s, ref=np.max) #applying short time fourier transform(STFT)

FREQ_BINS = N_FFT // 2 + 1

def get_cached_harmonic(song_id):
    cache_path = Path(CACHE_DIR) / f"harmonic_{song_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    apath = Path(AUDIO_DIR) / f"{song_id}.wav"
    audio, sr = librosa.load(str(apath), sr=SAMPLE_RATE, mono=True)
    feat = extract_harmonic(audio, sr) #normalizes the feature
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    np.save(cache_path, feat)
    return feat


# STEP 4 — DATA AUGMENTATION 


def augment_audio(audio, sr):
    aug = audio.copy()          #make copy of song
    if np.random.rand() < 0.5:  # random noise
        aug = aug + np.random.uniform(0.001, 0.005) * np.random.randn(len(aug))
    if np.random.rand() < 0.5: # Makes audio louder or quieter
        aug = aug * np.random.uniform(0.7, 1.3)
    if np.random.rand() < 0.3: #low pass filter
        if np.random.rand() < 0.5:
            cutoff = np.random.uniform(2000, 6000)
            b, a = scipy.signal.butter(4, cutoff / (sr/2), btype="low")
        else:
            cutoff = np.random.uniform(100, 500)
            b, a = scipy.signal.butter(4, cutoff / (sr/2), btype="high")
        aug = scipy.signal.filtfilt(b, a, aug).astype(np.float32)
    return aug.astype(np.float32)


# STEP 5 — DATASET (instant: center frame label + boundary label)


class InstantDataset(Dataset):
    def __init__(self, song_ids, chunk_frames, augment=True, n_aug=1):
        self.chunks = []
        self.eval_data = []
        self.chunk_frames = chunk_frames
        frame_dur = HOP_LENGTH / SAMPLE_RATE # 0.032 seconds per frame

        for sid in tqdm(song_ids, desc="Building dataset [harmonic-cnn]"):
            apath = Path(AUDIO_DIR) / f"{sid}.wav"
            if not apath.exists(): continue
            segs = load_annotation(sid)
            if not segs: continue
            try:
                feat = get_cached_harmonic(sid)
            except Exception:
                continue

            n = feat.shape[1]

            # function labels
            f_labels = np.full(n, LABEL_TO_ID["silence"], dtype=np.int64)
            for seg in segs: #For each segment, fills the corresponding frames with the correct label ID
                sf = int(seg["start"]/frame_dur)
                ef = min(int(seg["end"]/frame_dur), n)
                f_labels[sf:ef] = LABEL_TO_ID[seg["label"]]

            # boundary labels (0.6s around each boundary) everthing else is 0.0
            b_labels = np.zeros(n, dtype=np.float32)
            boundary_frames = int(0.6/frame_dur)
            for seg in segs:
                bf = int(seg["start"]/frame_dur)
                b_labels[max(0,bf-boundary_frames//2):min(n,bf+boundary_frames//2)] = 1.0

            self._add(feat, f_labels, b_labels, chunk_frames) #Cuts the song into 24-second chunks and stores each chunk with its function label and boundary labe
            self.eval_data.append((feat, segs, n*frame_dur)) # store after the training

            if augment:
                audio, sr = librosa.load(str(apath), sr=SAMPLE_RATE, mono=True)
                for _ in range(n_aug):
                    af = extract_harmonic(augment_audio(audio, sr), sr)
                    af = (af - af.mean())/(af.std()+1e-6) # normalization
                    nf = af.shape[1]
                    fl = f_labels[:nf] if nf<=len(f_labels) else \
                         np.pad(f_labels,(0,nf-len(f_labels)),constant_values=LABEL_TO_ID["silence"])
                    bl = b_labels[:nf] if nf<=len(b_labels) else \
                         np.pad(b_labels,(0,nf-len(b_labels)))
                    self._add(af, fl, bl, chunk_frames)

    def _add(self, feat, f_labels, b_labels, chunk_frames):#Creates many training chunks from each song.
        n = feat.shape[1]
        hop = max(chunk_frames//2, 1)
        for start in range(0, n-chunk_frames+1, hop):
            fc = feat[:, start:start+chunk_frames] #Slices the feature array to get one chunk. Shape: (513, chunk_frames).
            center = start + chunk_frames//2  #CENTER frame of this chunk
            self.chunks.append((
                th.tensor(fc, dtype=th.float32),
                th.tensor(f_labels[center], dtype=th.long),
                th.tensor(b_labels[center], dtype=th.float32)
            ))

    def __len__(self): return len(self.chunks)
    def __getitem__(self, i): return self.chunks[i]


# STEP 6 — HARMONIC-CNN MODEL ( 7 conv + 2 dense, 2 output heads)


class HarmonicCNN(nn.Module): #define neural networks
    def __init__(self, n_classes):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2) #2D covolution, normalizer, activation function, maxpool
            )
        # 7 conv layers maps 256 features 
        self.conv = nn.Sequential(
            block(1,   32),
            block(32,  64),
            block(64,  64),
            block(64,  128),
            block(128, 128),
            block(128, 256),
            block(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
                                         # 2 dense layers
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3)
        )
                                        # Two output heads
        self.function_head = nn.Linear(256, n_classes)  # 7 classes
        self.boundary_head = nn.Linear(256, 1)          # boundary prob

    def forward(self, x): # how dataflow form model
        if x.dim() == 3: x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.shared_fc(x)
        return self.function_head(x), self.boundary_head(x)


# STEP 7 — EVALUATION METRICS


def frames_to_segments(frame_labels, frame_dur):
    if len(frame_labels) == 0: return []
    segs, start = [], 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != frame_labels[i-1]:
            segs.append({"start":start*frame_dur,"end":i*frame_dur,
                         "label":ID_TO_LABEL[int(frame_labels[i-1])]}); start=i
    segs.append({"start":start*frame_dur,"end":len(frame_labels)*frame_dur,
                 "label":ID_TO_LABEL[int(frame_labels[-1])]})
    return [s for s in segs if s["end"]-s["start"]>0.01]

def to_mir(segments):
    b = np.array([s["start"] for s in segments]+[segments[-1]["end"]])
    ivs = np.column_stack([b[:-1], b[1:]])
    anns = [s["label"] for s in segments]
    keep = ivs[:,1]-ivs[:,0] > 0
    return ivs[keep], [a for a,k in zip(anns,keep) if k]

def compute_metrics(pred_segs, ref_segs):
    if not pred_segs or not ref_segs: return {k:0.0 for k in PAPER}
    pred_ivs, pred_ann = to_mir(pred_segs)
    ref_ivs,  ref_ann  = to_mir(ref_segs)
    if len(pred_ivs)==0 or len(ref_ivs)==0: return {k:0.0 for k in PAPER}
    _,_,hr5f = mir_eval.segment.detection(ref_ivs,pred_ivs,window=0.5,trim=True) #HR.5F
    fr=10.0; dur=max(ref_ivs[-1,1],pred_ivs[-1,1]); n=int(dur*fr)+1 
    def grid(segs):
        g=["silence"]*n
        for s in segs:
            a,b=int(s["start"]*fr),min(int(s["end"]*fr),n)
            for j in range(a,b): g[j]=s["label"]
        return g
    pg,rg=grid(pred_segs),grid(ref_segs)  # ACC
    acc=sum(p==r for p,r in zip(pg,rg))/n
    ce=min(ref_ivs[-1,1],pred_ivs[-1,1])
    def trim(ivs,anns,end):
        oi,oa=[],[]
        for (s,e),a in zip(ivs,anns):
            if s>=end: break
            oi.append([s,min(e,end)]); oa.append(a)
        return np.array(oi),oa
    ri,ra=trim(ref_ivs,ref_ann,ce); pi,pa=trim(pred_ivs,pred_ann,ce)
    if len(ri)==0 or len(pi)==0: pwf=sf=0.0
    else:
        ri[0,0]=pi[0,0]=0.0; ri[-1,1]=pi[-1,1]=ce
        _,_,pwf=mir_eval.segment.pairwise(ri,ra,pi,pa) #PWF
        _,_,sf =mir_eval.segment.nce(ri,ra,pi,pa)       #SF
    pc=[s for s in pred_segs if s["label"]=="chorus"]   #CHR.5F
    rc=[s for s in ref_segs  if s["label"]=="chorus"]
    if pc and rc:
        pcb=np.array([s["start"] for s in pc]+[pc[-1]["end"]]) #CF1
        rcb=np.array([s["start"] for s in rc]+[rc[-1]["end"]])
        _,_,chr5f=mir_eval.segment.detection(
            np.column_stack([rcb[:-1],rcb[1:]]),
            np.column_stack([pcb[:-1],pcb[1:]]),window=0.5,trim=True)
    else: chr5f=0.0
    pgb=["chorus" if x=="chorus" else "other" for x in pg]
    rgb=["chorus" if x=="chorus" else "other" for x in rg]
    def g2iv(g):
        ivs,anns,st=[],[],0
        for i in range(1,len(g)):
            if g[i]!=g[i-1]: ivs.append([st/fr,i/fr]);anns.append(g[i-1]);st=i
        ivs.append([st/fr,len(g)/fr]);anns.append(g[-1])
        return np.array(ivs),anns
    pv,pn=g2iv(pgb); rv,rn=g2iv(rgb)
    rv[0,0]=pv[0,0]=0.0; cm=min(rv[-1,1],pv[-1,1]); rv[-1,1]=pv[-1,1]=cm
    _,_,cf1=mir_eval.segment.pairwise(rv,rn,pv,pn)
    return {"HR.5F":hr5f,"ACC":acc,"PWF":pwf,"Sf":sf,"CHR.5F":chr5f,"CF1":cf1}

def evaluate(model, eval_data, chunk_frames):
    model.eval()
    frame_dur = HOP_LENGTH/SAMPLE_RATE
    metrics = []
    with th.no_grad():
        for feat, ref_segs, dur in eval_data:
            n = feat.shape[1]
            pred_frames = np.full(n, LABEL_TO_ID["silence"], dtype=np.int64)
            hop = max(chunk_frames//4, 1)
            for start in range(0, n-chunk_frames+1, hop):
                fc = feat[:, start:start+chunk_frames]
                x = th.tensor(fc, dtype=th.float32).unsqueeze(0).to(DEVICE)
                func, _ = model(x)
                lab = func.argmax(-1).item()
                center = start + chunk_frames//2
                pred_frames[max(0,center-hop//2):center+hop//2] = lab
            metrics.append(compute_metrics(frames_to_segments(pred_frames,frame_dur),ref_segs))
    return {k:float(np.mean([m[k] for m in metrics])) for k in PAPER}


# STEP 8 — TRAIN


def train(song_ids, epochs, batch_size, chunk_seconds):
    print(f"\n{'='*60}")
    print(f"  HARMONIC-CNN — Wang et al. (2022)")
    print(f"  Feature: Harmonic | Model: CNN (7 conv + 2 dense)")
    print(f"  Loss: 0.9 x boundary + 0.1 x function")
    print(f"  device: {DEVICE} | epochs: {epochs} | batch: {batch_size}")
    print(f"{'='*60}")

    frame_rate   = SAMPLE_RATE/HOP_LENGTH
    chunk_frames = int(chunk_seconds*frame_rate)
    chunk_frames = (chunk_frames//4)*4

    model = HarmonicCNN(N_CLASSES).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {params:,}")

    ds = InstantDataset(song_ids, chunk_frames, augment=True, n_aug=1) #total params 1232136 like that
    if len(ds)==0:
        print("  No chunks built."); return None
    print(f"  Dataset: {len(ds)} chunks (incl. augmented)") 

    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = th.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer. Updates all model weights during training. lr=0.0005 is the step size.
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=PATIENCE, factor=0.5)
    func_crit  = nn.CrossEntropyLoss() #Loss function for function labels. Measures how wrong the 7-class predictions are.
    bound_crit = nn.BCEWithLogitsLoss()# Loss function for boundary labels. Binary cross-entropy — measures how wrong the 0/1 boundary predictions are.

    best_loss = float("inf")
    model.train()

    for epoch in range(epochs):
        total_loss = total_b = total_f = 0
        n_batches = 0
        for feats, f_labels, b_labels in dl:
            feats    = feats.to(DEVICE)
            f_labels = f_labels.to(DEVICE)
            b_labels = b_labels.to(DEVICE)
            opt.zero_grad()
            func_out, bound_out = model(feats) #Forward pass — runs the batch through the model.
            f_loss = func_crit(func_out, f_labels) #Computes how wrong the function predictions are.
            b_loss = bound_crit(bound_out.squeeze(1), b_labels) #Computes how wrong the boundary predictions are. squeeze(1) removes the extra dimension.
            loss   = BOUNDARY_WEIGHT * b_loss + FUNCTION_WEIGHT * f_loss #loss — 0.9 × boundary + 0.1 × function
            loss.backward(); opt.step() #Backpropagation
            total_loss += loss.item(); total_f += f_loss.item(); total_b += b_loss.item()
            n_batches += 1
        avg = total_loss/n_batches
        scheduler.step(avg)
        if (epoch+1)%5==0 or epoch==0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={avg:.4f}  "
                  f"(boundary={total_b/n_batches:.4f}  "
                  f"function={total_f/n_batches:.4f})")
        if avg < best_loss:
            best_loss = avg
            th.save(model.state_dict(), "model_harmonic_cnn_best.pt")

    th.save(model.state_dict(), "model_harmonic_cnn_final.pt")
    print(f"  Final model saved.")

    model.load_state_dict(th.load("model_harmonic_cnn_best.pt"))
    print(f"  Evaluating with best model...")
    results = evaluate(model, ds.eval_data, chunk_frames)

    print(f"\n{'='*55}")
    print(f"  HARMONIC-CNN RESULTS vs PAPER")
    print(f"{'='*55}")
    print(f"  {'Metric':<10}{'Ours':>8}{'Paper':>8}{'Diff':>8}")
    print(f"  {'-'*34}")
    for k in PAPER:
        diff = results[k] - PAPER[k]
        print(f"  {k:<10}{results[k]:>8.3f}{PAPER[k]:>8.3f}{diff:>+8.3f}")
    print(f"{'='*55}")

    row = {"Model":"Harmonic-CNN", **{k:round(results[k],3) for k in PAPER}}
    pd.DataFrame([row]).to_csv(RESULTS_CSV, index=False)
    print(f"  Saved to {RESULTS_CSV}")
    return results


# MAIN


def get_song_ids(max_songs=None): #Defines a function that returns a list of valid song IDs. max_songs=None means by default return all songs.
    meta    = pd.read_csv(META_PATH) 
    popular = meta[meta["CLASS"]=="popular"]["SONG_ID"].astype(str).tolist()
    valid   = [s for s in popular
               if (Path(AUDIO_DIR)/f"{s}.wav").exists()
               and (Path(ANNOT_DIR)/s/"textfile1.txt").exists()]
    if max_songs: valid=valid[:max_songs]
    print(f"Found {len(valid)} usable songs")
    return valid

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["test","full"], default="test")
    args = p.parse_args()
    if args.mode == "test":
        songs=get_song_ids(max_songs=5); epochs=5; batch_size=4; chunk_sec=8
    else:
        songs=get_song_ids(max_songs=50); epochs=EPOCHS; batch_size=BATCH_SIZE; chunk_sec=CHUNK_SECONDS
    if not songs:
        print("No songs found."); sys.exit(1)
    train(songs, epochs, batch_size, chunk_sec)
