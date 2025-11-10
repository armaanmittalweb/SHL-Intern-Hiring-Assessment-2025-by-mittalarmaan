import os, gc, math, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import librosa
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

TRAIN_CSV = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/csvs/train.csv"
TEST_CSV  = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/csvs/test.csv"
TRAIN_AUDIO_DIR = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/audios/train"
TEST_AUDIO_DIR  = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset/audios/test"

OUT_DIR = Path("/kaggle/working/outputs_genai")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
EPOCHS = 14
PATIENCE = 5
BATCH = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
MAX_SECONDS = 15.0    
TTA_N = 5             
PROMPTS = 9           
HEAD_HIDDEN = 768     
HEAD_DROPOUT = 0.25
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
AMP = True

LSMOOTH = 0.05
W_CE = 0.55
W_KLD = 0.45
W_MSE = 0.35
W_CONS = 0.1
SIGMA_LDL = 0.5

KNN_K = 11
KNN_TEMP = 0.15
KNN_BLEND_OOF_SEARCH = True  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()


ALLOWED_EXTS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
def ensure_dot(ext): return ext if ext.startswith(".") else f".{ext}"

def file_for_id(id_value, base_dir) -> Path:
    base_dir = Path(base_dir); id_str = str(id_value)
    for ext in ALLOWED_EXTS:
        p = base_dir / f"{id_str}{ensure_dot(ext)}"
        if p.exists(): return p
    matches = list(base_dir.glob(f"{id_str}.*"))
    if matches: return matches[0]
    raise FileNotFoundError(f"Audio for id '{id_str}' not found in {base_dir}")

def infer_id_col(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for p in ["filename", "id", "uuid", "clip_id", "file_id"]:
        if p in lower_map: return lower_map[p]
    # fallback
    for c in df.columns:
        if df[c].dtype == object: return c
    return df.columns[0]

def infer_label_col(df: pd.DataFrame, id_col: str) -> str:
    num_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        for c in df.columns:
            if c != id_col: return c
    num_cols = sorted(num_cols, key=lambda c: df[c].nunique())
    return num_cols[0]

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
ID_COL = infer_id_col(train_df)
LABEL_COL = infer_label_col(train_df, ID_COL)
train_df[LABEL_COL] = train_df[LABEL_COL].astype(float)

label_values_sorted = sorted(train_df[LABEL_COL].dropna().unique().tolist())
n_classes = len(label_values_sorted)
label2idx = {v:i for i,v in enumerate(label_values_sorted)}
idx2label = {i:v for i,v in enumerate(label_values_sorted)}
train_df["y_idx"] = train_df[LABEL_COL].map(label2idx).astype(int)

print(f"Using device: {device}  (pin_memory={PIN_MEMORY}, workers={NUM_WORKERS})")
print(f"ID column: {ID_COL} (test uses: {ID_COL})")
print(f"Label column: {LABEL_COL}")
print(f"Classes ({n_classes}): {label_values_sorted}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for f, (_, va_idx) in enumerate(skf.split(train_df, train_df["y_idx"])):
    train_df.loc[va_idx, "fold"] = f
print("Fold sizes:", train_df["fold"].value_counts().sort_index().to_dict())


import torchaudio

def pick_ssl_bundle():
    candidates = []
    if hasattr(torchaudio.pipelines, "WAVLM_BASE_PLUS"):
        candidates.append(("WAVLM_BASE_PLUS", torchaudio.pipelines.WAVLM_BASE_PLUS))
    if hasattr(torchaudio.pipelines, "WAV2VEC2_BASE"):
        candidates.append(("WAV2VEC2_BASE", torchaudio.pipelines.WAV2VEC2_BASE))
    if hasattr(torchaudio.pipelines, "WAV2VEC2_ASR_BASE_960H"):
        candidates.append(("WAV2VEC2_ASR_BASE_960H", torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H))
    if not candidates:
        raise RuntimeError("No suitable torchaudio SSL bundle found.")
    return candidates[0]

SSL_NAME, SSL_BUNDLE = pick_ssl_bundle()
SSL_SR = SSL_BUNDLE.sample_rate
ssl_model = SSL_BUNDLE.get_model().to(device)
for p in ssl_model.parameters(): p.requires_grad = False
ssl_model.eval()
print(f"Backbone: {SSL_NAME} (sr={SSL_SR})")

try:
    HIDDEN_DIM = getattr(ssl_model, "encoder_embed_dim", None) \
                 or getattr(ssl_model, "encoder", None).transformer.layers[0].self_attn.model_dim
except Exception:
    HIDDEN_DIM = None

class WaveDataset(Dataset):
    def __init__(self, df, audio_dir, id_col, label_col=None, train=True, tta_random=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.id_col = id_col
        self.label_col = label_col
        self.train = train
        self.tta_random = tta_random
        self.max_len = int(MAX_SECONDS * SSL_SR)

    def __len__(self): return len(self.df)

    def _load_wave(self, wav_path: Path):
        y, _ = librosa.load(str(wav_path), sr=SSL_SR, mono=True)
        if y is None or len(y) == 0:
            y = np.zeros(self.max_len, dtype=np.float32)
        return y.astype(np.float32)

    def _crop_or_pad(self, y: np.ndarray, train: bool):
        L = len(y)
        if L >= self.max_len:
            if train or self.tta_random:
                start = random.randint(0, L - self.max_len)
            else:
                start = max(0, (L - self.max_len)//2)
            return y[start:start+self.max_len]
        else:
            out = np.zeros(self.max_len, dtype=np.float32)
            start = 0 if train or self.tta_random else (self.max_len - L)//2
            out[start:start+L] = y
            return out

    def _augment(self, y: np.ndarray):
        if random.random() < 0.5:
            gain = 10 ** (random.uniform(-3, 3) / 20)  
            y = y * gain
        if random.random() < 0.4:
            noise = np.random.randn(*y.shape).astype(np.float32) * random.uniform(0.001, 0.02)
            y = y + noise
        if random.random() < 0.4:
            shift = int(random.uniform(-0.02, 0.02) * len(y))
            y = np.roll(y, shift)
        y = np.clip(y, -1.0, 1.0)
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        _id = str(row[self.id_col])
        wav_path = file_for_id(_id, self.audio_dir)
        y = self._load_wave(wav_path)
        if self.train or self.tta_random:
            y = self._augment(y)
        y = self._crop_or_pad(y, train=(self.train or self.tta_random))
        x = torch.tensor(y, dtype=torch.float32)
        if self.label_col is not None:
            y_idx = int(row["y_idx"])
            y_num = float(row[self.label_col])
        else:
            y_idx, y_num = -1, -1.0
        return x, torch.tensor(y_idx), torch.tensor(y_num, dtype=torch.float32), _id

class PromptHead(nn.Module):
    def __init__(self, feat_dim, n_classes, n_prompts=9, n_heads=6, drop=0.25):
        super().__init__()
        self.n_classes = n_classes
        self.prompts = nn.Parameter(torch.randn(n_prompts, feat_dim) / math.sqrt(feat_dim))
        self.attn = nn.MultiheadAttention(feat_dim, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(feat_dim*2, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self.drop = nn.Dropout(drop)
        self.head_cls = nn.Linear(feat_dim, n_classes)
        self.head_reg = nn.Linear(feat_dim, 1)

    def forward(self, feats): 
        B, T, C = feats.shape
        q = self.prompts.unsqueeze(0).expand(B, -1, -1)   
        attn_out, _ = self.attn(q, feats, feats)         
        pooled = attn_out.mean(dim=1)                     
        pooled = self.ffn(pooled)
        pooled = self.drop(pooled)
        logits = self.head_cls(pooled)
        reg = self.head_reg(pooled).squeeze(1)
        return logits, reg, pooled


@torch.no_grad()
def ssl_extract(wave_batch: torch.Tensor):
    """
    wave_batch: [B, L] at SR=SSL_SR (float32)
    returns: [B, T, C]
    """
    ssl_model.eval()
    out = ssl_model.extract_features(wave_batch.to(device))[0]
    if isinstance(out, list):  
        x = out[-1]
    else:
        x = out
    return x  


class_values = torch.tensor(label_values_sorted, dtype=torch.float32, device=device)

def make_ldl_targets(y_float_t: torch.Tensor, sigma=SIGMA_LDL):
    d2 = (y_float_t.view(-1,1) - class_values.view(1,-1))**2
    w = torch.exp(-d2 / (2.0 * (sigma**2)))
    return w / (w.sum(dim=1, keepdim=True) + 1e-12)

ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=LSMOOTH)

def compute_losses(logits, reg, y_idx, y_num):
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    loss_ce = ce_loss_fn(logits, y_idx)
    y_dist = make_ldl_targets(y_num)
    loss_kld = torch.nn.functional.kl_div(log_probs, y_dist, reduction="batchmean")
    loss_mse = nn.functional.mse_loss(reg, y_num)
    ev_cls = (probs * class_values[None, :]).sum(dim=1)
    loss_cons = nn.functional.mse_loss(ev_cls, reg.detach())
    total = W_CE*loss_ce + W_KLD*loss_kld + W_MSE*loss_mse + W_CONS*loss_cons
    return total, ev_cls


def train_one_epoch(head, dl, optimizer, scaler):
    head.train()
    total_loss, all_pred_idx, all_true_idx = 0.0, [], []
    for xb, y_idx, y_num, _ids in dl:
        xb = xb.to(device, non_blocking=True)
        y_idx = y_idx.to(device, non_blocking=True)
        y_num = y_num.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP):
            feats = ssl_extract(xb)         
            logits, reg, pooled = head(feats)
            loss, ev_cls = compute_losses(logits, reg, y_idx, y_num)

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 2.0)
        scaler.step(optimizer); scaler.update()

        total_loss += float(loss.item()) * xb.size(0)
        with torch.no_grad():
            pred_idx = logits.argmax(1).detach().cpu().numpy()
            all_pred_idx.append(pred_idx)
            all_true_idx.append(y_idx.detach().cpu().numpy())

    all_pred_idx = np.concatenate(all_pred_idx); all_true_idx = np.concatenate(all_true_idx)
    acc = float((all_pred_idx == all_true_idx).mean())
    f1  = f1_score(all_true_idx, all_pred_idx, average="macro")
    return total_loss / len(dl.dataset), acc, f1

@torch.no_grad()
def eval_epoch(head, dl):
    head.eval()
    logits_list, regs_list, feats_list = [], [], []
    y_idx_true, y_num_true = [], []
    for xb, y_idx, y_num, _ids in dl:
        xb = xb.to(device, non_blocking=True)
        feats = ssl_extract(xb)
        logits, reg, pooled = head(feats)
        logits_list.append(logits.cpu()); regs_list.append(reg.cpu()); feats_list.append(pooled.cpu())
        y_idx_true.append(y_idx); y_num_true.append(y_num)

    logits = torch.cat(logits_list, 0).numpy()
    regs   = torch.cat(regs_list, 0).numpy().astype(np.float32)
    feats  = torch.cat(feats_list, 0).numpy().astype(np.float32)
    y_idx_true = torch.cat(y_idx_true, 0).numpy()
    y_num_true = torch.cat(y_num_true, 0).numpy().astype(np.float32)

    probs = sklearn.utils.extmath.softmax(logits, copy=False)
    pred_idx = probs.argmax(1)
    acc = float((pred_idx == y_idx_true).mean())
    f1  = f1_score(y_idx_true, pred_idx, average="macro")

    class_vals = np.array(label_values_sorted, dtype=np.float32)
    ev_cls = (probs * class_vals[None, :]).sum(1)
    rmse_cls = float(np.sqrt(np.mean((ev_cls - y_num_true)**2)))
    rmse_reg = float(np.sqrt(np.mean((regs   - y_num_true)**2)))
    rmse_ens = float(np.sqrt(np.mean(((0.5*ev_cls + 0.5*regs) - y_num_true)**2)))
    pearson = float(np.corrcoef(ev_cls, y_num_true)[0,1]) if (np.std(ev_cls)>1e-12 and np.std(y_num_true)>1e-12) else 0.0

    return {"probs": probs.astype(np.float32), "regs": regs, "feats": feats,
            "acc": acc, "f1": f1, "rmse_cls": rmse_cls, "rmse_reg": rmse_reg,
            "rmse_ens": rmse_ens, "pearson": pearson}

@torch.no_grad()
def predict_test(head, base_df, audio_dir, id_col, tta_n=TTA_N):
    head.eval()
    ds_center = WaveDataset(base_df, audio_dir, id_col, label_col=None, train=False, tta_random=False)
    dl_center = DataLoader(ds_center, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    probs, regs, feats, all_ids = [], [], [], []
    for xb, yi, yn, _ids in dl_center:
        xb = xb.to(device, non_blocking=True)
        f = ssl_extract(xb)
        logits, reg, pooled = head(f)
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu().numpy()); regs.append(reg.cpu().numpy()); feats.append(pooled.cpu().numpy())
        all_ids += list(_ids)
    probs = np.concatenate(probs); regs = np.concatenate(regs).astype(np.float32); feats = np.concatenate(feats)

    for _ in range(max(0, tta_n-1)):
        ds_rand = WaveDataset(base_df, audio_dir, id_col, label_col=None, train=False, tta_random=True)
        dl_rand = DataLoader(ds_rand, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        p_list, r_list, f_list = [], [], []
        for xb, yi, yn, _ids in dl_rand:
            xb = xb.to(device, non_blocking=True)
            f = ssl_extract(xb)
            logits, reg, pooled = head(f)
            p_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            r_list.append(reg.cpu().numpy()); f_list.append(pooled.cpu().numpy())
        probs += np.concatenate(p_list); regs += np.concatenate(r_list); feats += np.concatenate(f_list)
    probs /= tta_n; regs /= tta_n; feats /= tta_n
    return probs, regs, feats, all_ids


oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
oof_regs  = np.zeros((len(train_df),), dtype=np.float32)
oof_feats = None
val_indices_by_fold = {}
models = []

for fold in range(N_FOLDS):
    print(f"\n========== Fold {fold+1}/{N_FOLDS} ({SSL_NAME} + prompt head) ==========")
    trn = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val = train_df[train_df["fold"] == fold].reset_index(drop=True)
    val_indices_by_fold[fold] = train_df.index[train_df["fold"] == fold].values

    ds_trn = WaveDataset(trn, TRAIN_AUDIO_DIR, ID_COL, LABEL_COL, train=True, tta_random=False)
    ds_val = WaveDataset(val, TRAIN_AUDIO_DIR, ID_COL, LABEL_COL, train=False, tta_random=False)

    dl_trn = DataLoader(ds_trn, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if 'HIDDEN_DIM' not in globals() or HIDDEN_DIM is None:
        xb0, *_ = next(iter(dl_trn))
        with torch.no_grad():
            xfeat = ssl_extract(xb0.to(device))
        HIDDEN_DIM = xfeat.shape[-1]
        print("Inferred backbone dim:", HIDDEN_DIM)

    head = PromptHead(feat_dim=HIDDEN_DIM, n_classes=n_classes, n_prompts=PROMPTS,
                      n_heads=6, drop=HEAD_DROPOUT).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS-1))
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_rmse = float("inf"); best_state = None; patience = 0

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS} — lr={optimizer.param_groups[0]['lr']:.3e}")
        tr_loss, tr_acc, tr_f1 = train_one_epoch(head, dl_trn, optimizer, scaler)
        scheduler.step()
        res = eval_epoch(head, dl_val)
        print(f"train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
              f"valid: acc={res['acc']:.4f} f1={res['f1']:.4f} rmse(cls/reg/ens)={res['rmse_cls']:.4f}/{res['rmse_reg']:.4f}/{res['rmse_ens']:.4f} pearson={res['pearson']:.4f}")

        if res["rmse_ens"] + 1e-6 < best_rmse:
            best_rmse = res["rmse_ens"]; best_state = head.state_dict(); patience = 0
            print(f"  ↳ Saved best head (val RMSE(ens)={best_rmse:.4f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  ↳ Early stopping.")
                break
        gc.collect(); torch.cuda.empty_cache()

    head.load_state_dict(best_state); head.eval()
    res_val = eval_epoch(head, dl_val)
    print(f"Fold {fold} — VAL: acc={res_val['acc']:.4f} f1={res_val['f1']:.4f} | "
          f"RMSE cls/reg/ens = {res_val['rmse_cls']:.4f}/{res_val['rmse_reg']:.4f}/{res_val['rmse_ens']:.4f} | "
          f"Pearson={res_val['pearson']:.4f}")

    va_idx = val_indices_by_fold[fold]
    oof_probs[va_idx] = res_val["probs"]
    oof_regs[va_idx]  = res_val["regs"]
    if oof_feats is None:
        oof_feats = np.zeros((len(train_df), res_val["feats"].shape[1]), dtype=np.float32)
    oof_feats[va_idx] = res_val["feats"].astype(np.float32)

    models.append(best_state)


true_idx = train_df["y_idx"].values
true_num = train_df[LABEL_COL].values.astype(np.float32)
class_vals = np.array(label_values_sorted, dtype=np.float32)
oof_ev_cls = (oof_probs * class_vals[None, :]).sum(1)
oof_idx = oof_probs.argmax(1)
oof_acc = float((oof_idx == true_idx).mean())
oof_f1 = f1_score(true_idx, oof_idx, average="macro")
rmse_cls = float(np.sqrt(np.mean((oof_ev_cls - true_num)**2)))
rmse_reg = float(np.sqrt(np.mean((oof_regs   - true_num)**2)))
rmse_ens = float(np.sqrt(np.mean(((0.5*oof_ev_cls + 0.5*oof_regs) - true_num)**2)))
pearson  = float(np.corrcoef(oof_ev_cls, true_num)[0,1]) if (np.std(oof_ev_cls)>1e-12 and np.std(true_num)>1e-12) else 0.0

print("\n==== OOF METRICS (raw) ====")
print(f"Classification: acc={oof_acc:.4f} macroF1={oof_f1:.4f} | RMSE={rmse_cls:.4f} | Pearson r={pearson:.4f}")
print(f"Regression:                          RMSE={rmse_reg:.4f}")
print(f"Ensemble (0.5*cls + 0.5*reg):        RMSE={rmse_ens:.4f}")

fold_isos = {}
oof_ev_iso = np.zeros_like(oof_ev_cls)
for f in range(N_FOLDS):
    tr_mask = (train_df["fold"].values != f)
    va_mask = (train_df["fold"].values == f)
    iso = IsotonicRegression(y_min=class_vals.min(), y_max=class_vals.max(),
                             increasing=True, out_of_bounds="clip")
    iso.fit(oof_ev_cls[tr_mask], true_num[tr_mask])
    oof_ev_iso[va_mask] = iso.transform(oof_ev_cls[va_mask])
    fold_isos[f] = iso

rmse_iso = float(np.sqrt(np.mean((oof_ev_iso - true_num)**2)))
print(f"Isotonic OOF RMSE={rmse_iso:.4f} (vs EV {rmse_cls:.4f})")
USE_ISO = rmse_iso + 1e-6 < rmse_cls
print("Use isotonic:", USE_ISO)

nbrs = NearestNeighbors(n_neighbors=min(KNN_K, len(oof_feats)), metric="cosine")
nbrs.fit(oof_feats)

def knn_predict(feats: np.ndarray, labels: np.ndarray, query_feats: np.ndarray,
                k=KNN_K, temp=KNN_TEMP):
    k = min(k, len(feats))
    dists, idxs = nbrs.kneighbors(query_feats, n_neighbors=k, return_distance=True)
    sims = 1.0 - dists  
    weights = np.exp(sims / max(1e-6, temp))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-9)
    pred = (labels[idxs] * weights).sum(axis=1)
    return pred.astype(np.float32)

oof_knn = knn_predict(oof_feats, true_num, oof_feats, k=KNN_K, temp=KNN_TEMP)
rmse_knn = float(np.sqrt(np.mean((oof_knn - true_num)**2)))
print(f"kNN (embedding) OOF RMSE={rmse_knn:.4f}")

ev_base = oof_ev_iso if USE_ISO else oof_ev_cls
best = (None, 1e9)
grid = np.linspace(0, 1, 11)
for a in grid:          
    for b in grid:      
        for c in grid:  
            if a + b + c <= 1.0:
                d = 1.0 - a - b - c
                pred = a*ev_base + b*oof_regs + c*oof_knn + d*(0.5*ev_base + 0.5*oof_regs)  
                rmse = float(np.sqrt(np.mean((pred - true_num)**2)))
                if rmse < best[1]:
                    best = ((a,b,c,d), rmse)
print(f"Best OOF blend weights (ev, reg, knn, ens)={best[0]} → RMSE={best[1]:.4f}")
W_EV, W_REG, W_KNN, W_ENS = best[0]


test_probs_folds, test_regs_folds, test_feats_folds = [], [], []
for fold in range(N_FOLDS):
    head = PromptHead(HIDDEN_DIM, n_classes, n_prompts=PROMPTS, n_heads=6, drop=HEAD_DROPOUT).to(device)
    head.load_state_dict(models[fold]); head.eval()
    probs, regs, feats, ids = predict_test(head, test_df, TEST_AUDIO_DIR, ID_COL, tta_n=TTA_N)
    test_probs_folds.append(probs); test_regs_folds.append(regs); test_feats_folds.append(feats)
    del head; gc.collect(); torch.cuda.empty_cache()

test_probs = np.mean(test_probs_folds, axis=0)
test_regs  = np.mean(test_regs_folds,  axis=0)
test_feats = np.mean(test_feats_folds, axis=0)

class_vals = np.array(label_values_sorted, dtype=np.float32)
test_ev = (test_probs * class_vals[None, :]).sum(1)
if USE_ISO:
    test_ev_iso = np.zeros_like(test_ev)
    for f in range(N_FOLDS):
        ev_f = (test_probs_folds[f] * class_vals[None, :]).sum(1)
        test_ev_iso += fold_isos[f].transform(ev_f)
    test_ev_iso /= N_FOLDS
else:
    test_ev_iso = test_ev

test_knn = knn_predict(oof_feats, true_num, test_feats, k=KNN_K, temp=KNN_TEMP)

ev_base_test = test_ev_iso if USE_ISO else test_ev
test_pred_num = W_EV*ev_base_test + W_REG*test_regs + W_KNN*test_knn + W_ENS*(0.5*ev_base_test + 0.5*test_regs)

def round_to_half(x): return np.round(x * 2.0) / 2.0
MIN_LABEL, MAX_LABEL = float(min(label_values_sorted)), float(max(label_values_sorted))
test_pred_num = np.clip(round_to_half(test_pred_num), MIN_LABEL, MAX_LABEL)

def find_sample_submission(test_csv_path: str):
    p = Path(test_csv_path)
    cands = [
        p.with_name("sample_submission.csv"),
        p.parent / "sample_submission.csv",
        p.parent.parent / "csvs" / "sample_submission.csv",
        Path(TRAIN_CSV).with_name("sample_submission.csv"),
    ]
    for c in cands:
        if c.exists():
            try: return pd.read_csv(c)
            except: pass
    return None

sample_sub = find_sample_submission(TEST_CSV)
if sample_sub is not None and sample_sub.shape[1] >= 2:
    sub_id_col = sample_sub.columns[0]
    sub_label_col = sample_sub.columns[1]
    sub = pd.DataFrame({sub_id_col: pd.Series(test_df[ID_COL].astype(str).values, dtype=object),
                        sub_label_col: test_pred_num})
else:
    sub = pd.DataFrame({ID_COL: pd.Series(test_df[ID_COL].astype(str).values, dtype=object),
                        LABEL_COL: test_pred_num})

sub_path = OUT_DIR / "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote submission to: {sub_path}")
print(sub.head(10))
