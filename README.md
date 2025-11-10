# SSL Audio Regression–Classification with PromptHead

> A clean, well‑documented starter for continuous‑label audio prediction on **Kaggle** using a frozen self‑supervised speech backbone (WavLM/Wav2Vec2), an attention‑prompted head, multi‑objective training, calibrated predictions, and test‑time ensembling.

---

## ✨ Highlights

* **Plug‑and‑play** with SHL Intern Hiring Assessment 2025 dataset paths
* **Self‑supervised backbone** (torchaudio pipelines: WavLM / Wav2Vec2) kept **frozen** for speed & stability
* **PromptHead**: learnable prompts + Multi‑Head Attention → pooled embedding → **dual heads** (classification & regression)
* **Label Distribution Learning (LDL)** target + **CE+KLD+MSE+Consistency** multi‑loss
* **5‑fold CV**, **TTA** (center + random crops), **isotonic calibration**, **kNN on embeddings**, and **grid‑searched blending**
* **Auto‑ID/label inference** from CSV schema; **robust audio loader** with crop/pad & light augmentations
* **Single‑file** training + inference → writes `outputs_genai/submission.csv`

---

## 🗺️ Problem & Approach

**Task.** Predict a *continuous* target (e.g., rating in 0.5 steps) from single‑channel audio clips.

**Strategy.**

1. Turn the regression into **two views**:

   * **Classification over discrete label set** (e.g., {0.0, 0.5, …}) with expected value used as a numeric prediction.
   * **Direct regression** from pooled features.
2. Train a **frozen SSL encoder** to extract robust frame‑level features, then learn a compact **PromptHead**.
3. At validation/inference, combine: **EV(cls)**, **reg**, **kNN(embedding)**, and **EV/reg ensemble** with **grid‑searched weights**; optionally apply **isotonic calibration** learned out‑of‑fold.

---

## 🔧 Project Structure (single script)

```
train.csv / test.csv
train/  test/      # audio folders
└── <id>.wav|mp3|flac|ogg|m4a

main.py            # the code below (single file)
outputs_genai/     # artifacts, including submission.csv
```

---

## 🧱 End‑to‑End Pipeline

```mermaid
flowchart LR
    A[CSVs: train/test] --> B[Auto infer ID & label columns]
    B --> C[Stratified KFold (5)]
    C --> D[WaveDataset + Augment + Crop/Pad]
    D --> E[SSL Backbone (frozen)\nWavLM/W2V2 -> [B,T,C]]
    E --> F[PromptHead\n(prompts + MHA + FFN)]
    F --> G1[Head_cls -> logits]
    F --> G2[Head_reg -> scalar]
    G1 --> H1[CE + KLD (LDL)]
    G2 --> H2[MSE]
    G1--EV-->J[Consistency\n(EV vs Reg)]
    H1 --> I[Weighted multi‑loss]
    H2 --> I
    J --> I
    I --> K[Early‑stopped model per fold]
    K --> L[OOF: probs/reg/feats]
    L --> M[Isotonic calib (per fold)]
    L --> N[kNN on pooled feats]
    M --> O[Blend: EV, Reg, kNN, (EV+Reg)/2]
    N --> O
    O --> P[Test‑time TTA + fold‑avg]
    P --> Q[Round to 0.5 & clip]
    Q --> R[submission.csv]
```

---

## 🧠 Model Architecture

### 1) SSL Backbone (frozen)

* Picked via `torchaudio.pipelines` auto‑probe (prefers `WAVLM_BASE_PLUS`, falls back to `WAV2VEC2_BASE` / `WAV2VEC2_ASR_BASE_960H`).
* Inputs: mono float32 wave @ backbone SR; Outputs: `[B, T, C]` frame embeddings.

### 2) **PromptHead** (trainable)

* Parameters: `n_prompts = 9`, `n_heads = 6`, hidden dim = backbone C
* Learnable **prompt tokens** `P ∈ R^{n_prompts×C}` query the sequence via **Multi‑Head Attention**:

```mermaid
flowchart TB
    subgraph Sequence
    X1[feat t1] --> X2[feat t2] --> X3[...]
    end
    subgraph Prompts (learned)
    P1((p1))
    P2((p2))
    Pn((p9))
    end
    P1 -- Q,K,V=feats --> ATTN[MHA]
    P2 -- Q,K,V=feats --> ATTN
    Pn -- Q,K,V=feats --> ATTN
    ATTN --> Pool[mean over prompts]
    Pool --> FFN[LayerNorm→Linear→ReLU→Dropout→Linear→LayerNorm]
    FFN --> Heads
    subgraph Heads
    CLS[Linear→n_classes]
    REG[Linear→1]
    end
```

* **Outputs**: logits (classification), scalar (regression), pooled embedding (for kNN).

---

## 🎯 Training Objective (multi‑loss)

Let `logits` be class scores, `reg` regression output, `y_idx` the class index, `y_num` the numeric label, and `\{c_j\}` the sorted class values.

1. **Cross‑Entropy (CE)** with label smoothing `ε=LSMOOTH` on `y_idx`.
2. **Kullback–Leibler (KLD)** vs a **Label‑Distribution Learning** target:

   [ w_j \propto \exp\left(-\frac{(y_{num}-c_j)^2}{2\sigma^2}\right),\quad p^{\text{LDL}}=\frac{w}{\sum w} ]
3. **MSE (regression)**: `MSE(reg, y_num)`.
4. **Consistency**: `MSE( EV(probs, c_j), stop_grad(reg) )` where `EV=Σ p_j c_j`.

**Total**
[ \mathcal{L} = W_{CE}·CE + W_{KLD}·KLD + W_{MSE}·MSE + W_{CONS}·MSE_{cons} ]

With defaults: `W_CE=0.55`, `W_KLD=0.45`, `W_MSE=0.35`, `W_CONS=0.10`, `σ=0.5`.

---

## 📦 Data & Augmentation

* **Auto‑ID/label inference** from CSV headers (falls back sensibly).
* **Audio loading** via `librosa.load(..., sr=SSL_SR, mono=True)`.
* **Crop/Pad** to `MAX_SECONDS=15` at backbone sample rate.
* **Light augmentations** (train/tta): random gain, Gaussian noise, circular shift; values are clipped to `[-1,1]`.
* **TTA**: one center crop + `TTA_N-1` random crops; average probs, regs, and pooled embeddings.

---

## 🔁 Cross‑Validation & Early Stopping

* **StratifiedKFold (5)** on class indices.
* **Optimizer**: AdamW (lr=5e-4, wd=1e-4) with **CosineAnnealingLR**.
* **AMP** mixed precision, grad‑clip=2.0.
* **Early stop** by best **validation RMSE** on the 50/50 ensemble `(EV+reg)/2` with `PATIENCE=5`.

---

## 📏 Metrics & Calibration

* **Classification accuracy & Macro‑F1** from argmax of logits.
* **Regression RMSE** from `reg`.
* **EV RMSE** from expectation over class values.
* **Pearson r** between EV and labels.
* **Isotonic Regression**: fit per‑fold on OOF `(EV → y)`; apply fold‑wise to test EV; used if it **improves OOF RMSE**.

---

## 🧭 Embedding kNN & Blending

* Fit **NearestNeighbors(cosine)** on **OOF pooled embeddings**.
* Predict numeric label for a query embedding by temperature‑weighted neighbor average (`K=11`, `temp=0.15`).
* **Grid search** `a,b,c,d ∈ {0,0.1,…,1}` with `a+b+c+d=1` over:

  * `a·EV + b·reg + c·kNN + d·(EV+reg)/2` (with EV optionally isotonic‑calibrated)
* Pick weights with **best OOF RMSE**; reuse for test.

---

## 🚀 Inference & Submission

* Average **fold models** (by predictions), apply **TTA**.
* Compute `test_ev` → optional per‑fold isotonic → `test_knn` → **blend** with best weights.
* **Round to nearest 0.5**, **clip** to min/max label, and write `outputs_genai/submission.csv`.

---

## ⚙️ Configuration (key defaults)

| Param          | Default | Meaning                                      |
| -------------- | ------: | -------------------------------------------- |
| `N_FOLDS`      |       5 | Stratified K‑folds                           |
| `EPOCHS`       |      14 | Max epochs per fold                          |
| `PATIENCE`     |       5 | Early stop on val RMSE                       |
| `BATCH`        |      16 | Global batch size                            |
| `LR`           |    5e‑4 | AdamW learning rate                          |
| `MAX_SECONDS`  |    15.0 | Crop/pad duration                            |
| `TTA_N`        |       5 | Num crops at test                            |
| `PROMPTS`      |       9 | Prompt tokens in head                        |
| `HEAD_HIDDEN`  |     768 | Head hidden (unused; inferred from backbone) |
| `HEAD_DROPOUT` |    0.25 | Dropout in FFN/head                          |
| `LSMOOTH`      |    0.05 | CE label smoothing                           |
| `SIGMA_LDL`    |     0.5 | Gaussian width for LDL                       |
| `KNN_K`        |      11 | #neighbors for kNN                           |
| `KNN_TEMP`     |    0.15 | Softmax temperature for kNN                  |

> **Tip**: Paths are pre‑set for Kaggle input mounts; change `TRAIN_CSV`, `TEST_CSV`, `TRAIN_AUDIO_DIR`, `TEST_AUDIO_DIR` for local runs.

---

## 🧪 Quickstart (Kaggle Notebook)

1. **Ensure dataset paths** match the competition’s input structure.
2. **Run all** — the script will:

   * Detect SSL backbone & sample rate
   * Build folds → train heads → save best per fold
   * Compute OOF metrics, isotonic, kNN, and blending weights
   * Predict test with TTA → write `outputs_genai/submission.csv`

### Local Run (conda)

```bash
conda create -n audio python=3.10 -y
conda activate audio
pip install numpy pandas scikit-learn torch torchaudio librosa
python main.py
```

> If CUDA is available, AMP and `pin_memory=True` are enabled automatically.

---

## 🔍 Implementation Notes

* **Backbone dim discovery**: If not detectable, we infer `C` by a forward pass from one batch.
* **Robust file lookup**: supports `wav/mp3/flac/ogg/m4a`; picks the first match if multiple.
* **Augmentations** are intentionally light (safe for speech/music); tune probabilities for your domain.
* **Numerical stability**: softmax/exp guarded with eps; consistency term uses `detach()` to avoid degenerate coupling.
* **Reproducibility**: global seeds set; note that `DataLoader` shuffling and CUDA kernels can still introduce minor nondeterminism.

---

## 🧰 Troubleshooting

* **“No suitable torchaudio SSL bundle found.”**

  * Your torchaudio build may lack pretrained pipelines. Upgrade `torchaudio` or switch to a version with pipelines.
* **OOM (Out of Memory)**

  * Lower `BATCH`, `MAX_SECONDS`, or `TTA_N`; disable AMP if on CPU.
* **Mismatched IDs/labels**

  * Inspect `infer_id_col` / `infer_label_col` logic; override by setting `ID_COL`/`LABEL_COL` manually after CSV read.
* **Silent audios**

  * Loader replaces empty/invalid audio with zeros to keep batch shapes consistent.

---

## 🧪 Ablation Ideas

* **Unfreeze last N backbone layers** with a small LR for extra accuracy.
* Replace LDL Gaussian `σ` or use **ordinal regression** losses.
* Swap **PromptHead pooling** (mean→attentive pooling across prompts) or vary prompt count.
* Try **SpecAugment‑style** time masks on waveform or embeddings.
* Replace kNN with **ridge regression** on embeddings for meta‑learning.

---

## 📜 License & Attribution

* Backbones & pipelines provided by **torchaudio**. See their licenses.
* This README and training script released under MIT unless your competition rules impose otherwise.

---

## 🧾 Appendix: Key Code Snippets

<details>
<summary><strong>PromptHead</strong></summary>

```python
class PromptHead(nn.Module):
    def __init__(self, feat_dim, n_classes, n_prompts=9, n_heads=6, drop=0.25):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(n_prompts, feat_dim)/math.sqrt(feat_dim))
        self.attn = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim*2), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(feat_dim*2, feat_dim), nn.LayerNorm(feat_dim),
        )
        self.drop = nn.Dropout(drop)
        self.head_cls = nn.Linear(feat_dim, n_classes)
        self.head_reg = nn.Linear(feat_dim, 1)
    def forward(self, feats):
        B, T, C = feats.shape
        q = self.prompts.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.attn(q, feats, feats)
        pooled = self.ffn(attn_out.mean(dim=1))
        pooled = self.drop(pooled)
        return self.head_cls(pooled), self.head_reg(pooled).squeeze(1), pooled
```

</details>

<details>
<summary><strong>LDL target</strong></summary>

```python
def make_ldl_targets(y_float_t, sigma):
    d2 = (y_float_t.view(-1,1) - class_values.view(1,-1))**2
    w = torch.exp(-d2/(2*sigma**2))
    return w / (w.sum(dim=1, keepdim=True) + 1e-12)
```

</details>

<details>
<summary><strong>kNN over embeddings (cosine + temperature)</strong></summary>

```python
def knn_predict(feats, labels, query_feats, k=11, temp=0.15):
    dists, idxs = nbrs.kneighbors(query_feats, n_neighbors=min(k, len(feats)), return_distance=True)
    sims = 1.0 - dists
    weights = np.exp(sims / max(1e-6, temp))
    weights /= (weights.sum(axis=1, keepdims=True) + 1e-9)
    return (labels[idxs]*weights).sum(axis=1).astype(np.float32)
```

</details>

---

## 🙌 Acknowledgements

* Inspired by recent prompt‑pooling and attentive pooling ideas for audio.
* Thanks to the Kaggle community for standard CV/TTA/calibration/blending recipes.
