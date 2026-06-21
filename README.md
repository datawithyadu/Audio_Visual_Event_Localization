# Audio-Visual Event Localization (AVE)

Per-second audio-visual event localization on the AVE dataset using a BiLSTM attention network (att_Net).  
Given a 10-second video clip, the model predicts the event category for every second — 28 event categories + Background.

> Based on: **Tian et al., "Audio-Visual Event Localization in Unconstrained Videos", ECCV 2018**

---

## Results

Trained and evaluated on the official pre-extracted features released by Tian et al.

| Metric | Score |
|--------|-------|
| Per-second accuracy | 68.4% |
| **Macro recall** (primary metric) | **80.1%** |
| Mean Temporal IoU | 80.0% |
| Clips with IoU ≥ 0.5 | 80.9% |

Full per-class breakdown: [`results.txt`](results.txt)

---

## Dataset

- **AVE dataset**: 4,143 video clips × 10 seconds, 28 event categories + Background = 29 classes
- Download videos: [Google Drive](https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK)
- Download pre-extracted features (7.7 GB): released alongside the paper
- Feature extraction scripts (original): [Google Drive](https://drive.google.com/file/d/1TJL3cIpZsPHGVAdMgyr43u_vlsxcghKY/view?usp=sharing)
- Annotations: `Category & VideoID & Quality & StartTime & EndTime` per line
- Splits: train (3,339) / val (402) / test (402)

### Folder structure expected

```
Audio_visual_event_localization/
├── data/                       ← all data lives here (gitignored)
│   ├── audio_feature.h5        ← official VGGish features (42 MB)
│   ├── visual_feature.h5       ← official VGG19 features (7.7 GB)
│   ├── Annotations.txt
│   ├── trainSet.txt
│   ├── valSet.txt
│   ├── testSet.txt
│   ├── AVE/                    ← video .mp4 files (only for custom extraction)
│   └── features/               ← custom-extracted .pt files (auto-generated)
├── checkpoints/
├── config.py
├── ...
```

> Mirrors the original AVE repo's `/data/` convention, adapted as a relative path so it works on both Windows and Linux.

---

## Architecture

```
Audio (10 × 128)                Video (10 × 7 × 7 × 512)
[VGGish features]               [VGG19 pool5 features]
       │                                    │
       │                         reshape → (10, 49, 512)
       │                                    │
       └──────── AudioGuidedAttention ──────┘
                         │
                attended video (10, 512)
                         │
         ┌───────────────┴───────────────┐
    BiLSTM (audio)                  BiLSTM (video)
    128 → 256                       512 → 512
         │                              │
         └──────────── concat ──────────┘
                         │
                    768 → fc(256) → fc(29)
                         │
                per-second logits (10, 29)
```

**Key design choices**
- **Audio-guided attention**: audio embedding selects relevant spatial regions from 49 video patches
- **Modality dropout** (p=0.1): randomly zeros one modality per batch item to prevent dominance
- **Class-weighted CrossEntropyLoss**: inverse-frequency weights handle 29-class imbalance
- **Gradient clipping** at max norm 5.0 for BiLSTM stability

---

## Project Structure

```
Audio_visual_event_localization/
├── config.py               # All hyperparameters, paths, and constants
├── utils.py                # Annotation parsing, label building, class weights
├── dataset.py              # AVEDataset (.pt files) + AVEDatasetH5 (h5 files)
├── models.py               # AVEModel, AudioGuidedAttention, R2Plus1DEncoder
├── feature_extractor.py    # Custom VGGish + R(2+1)D feature extraction
├── train.py                # Training loop utilities
├── evaluate.py             # Accuracy, recall, temporal IoU metrics
├── run_pipeline.py         # End-to-end: extract → train → evaluate
│
├── feature_extractor/      # Original feature extraction scripts (Tian et al.)
│   ├── audio_feature_extractor.py
│   └── visual_feature_extractor.py
│
├── checkpoints/
│   └── best_model.pt       # Best checkpoint (saved by val loss)
└── results.txt             # Final test-set evaluation report
```

---

## Setup

```bash
conda create -n torch_env python=3.10
conda activate torch_env

# PyTorch with CUDA 12.4 (adjust index URL for your CUDA version)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

pip install numpy h5py librosa opencv-python scikit-learn resampy matplotlib
```

**Requirements**: NVIDIA GPU with CUDA, `audio_feature.h5` and `visual_feature.h5` placed in the `data/` folder.

---

## Run

```bash
conda activate torch_env
python run_pipeline.py
```

The pipeline has 3 stages that run automatically:

| Stage | Description | Time |
|-------|-------------|------|
| 1 | Feature extraction — **skipped automatically** if h5 files are present | Instant |
| 2 | Training — 30 epochs max, early stopping (patience=5) | ~8 min |
| 3 | Evaluation — accuracy, recall, temporal IoU, per-class report | ~1 min |

**If h5 files are present** (recommended): Stage 1 is skipped, training starts immediately using `AVEDatasetH5`.  
**If h5 files are absent**: Stage 1 extracts features from raw `.mp4` videos using VGGish + R(2+1)D and saves `.pt` files.

Live progress: `Get-Content pipeline.log -Wait -Tail 30`

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Dropout | 0.3 |
| Modality dropout | 0.1 |
| LSTM hidden (audio BiLSTM) | 128 → 256 |
| LSTM hidden (video BiLSTM) | 256 → 512 |

---

## Metrics

- **Macro recall** — average recall across 28 event classes, Background excluded *(primary metric)*
- **Temporal IoU** — intersection-over-union of predicted vs ground-truth event windows per clip
- **Per-second accuracy** — fraction of seconds correctly classified

### Baseline comparison

| Baseline | Recall |
|----------|--------|
| Always predict Background | 0.0% |
| Random uniform (29 classes) | ~3.4% |
| Custom extraction (VGGish + R(2+1)D) | 72.8% |
| **Official h5 features (VGG19) — this repo** | **80.1%** |

---

## Citation

If you use the AVE dataset, please cite:

```bibtex
@inproceedings{TianECCV2018,
    title={Audio-Visual Event Localization in Unconstrained Videos},
    author={Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
    booktitle={Computer Vision -- ECCV 2018},
    year={2018},
    publisher={Springer}
}
```

---

## GPU

Tested on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.4 / driver 591.74.
