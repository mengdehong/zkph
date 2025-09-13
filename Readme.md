# TRUSTWORTHY AND PRIVACY-PRESERVING PERCEPTUAL HASHING WITH ZERO-KNOWLEDGE PROOFS

This repository provides a reference implementation of the paper “Trustworthy and Privacy-Preserving Perceptual Hashing with Zero-Knowledge Proofs”, including:
- Perceptual hashing deep models (training and evaluation, PyTorch)
- A zero-knowledge proof prototype for Hamming distance based on multilinear sum-check and polynomial commitments (Rust, arkworks)

We ship a Docker image for a reproducible environment and provide scripts for experiments and benchmarking.


## Table of Contents
- Overview and Contributions
- Repository Structure
- Environment and Dependencies
  - Docker (recommended)
  - Local Installation (optional)
- Data Preparation
- Training and Evaluation
  - Training (TrainAll)
  - PIHD metrics (AUC, FPR@95TPR)
  - COCO Zero-shot Generalization
  - Ablation Study
- ZK Proof Benchmark (HDProof)
- Reproducibility Checklist and Expected Artifacts
- FAQ
- Acknowledgements
- Citation


## Overview and Contributions
This implementation targets trustworthy and privacy-preserving perceptual hashing with a unified training and verification pipeline:
- Multiple backbones (ResNet50, ViT, ConvNeXtV2, Swin-T, MambaOut, GroupMamba) with Baseline/Enhanced strategies
- Unified training and evaluation producing ROC-AUC and FPR@95TPR
- Zero-shot generalization evaluation on COCO
- A Hamming-distance ZK proof (hamproof) prototype and benchmark


## Repository Structure
```
PerceptHash/                 # Perceptual hashing models and scripts (PyTorch)
  model/                     # Models and backbones, includes kernels/selective_scan extension
  eval/                      # Evaluation scripts (AUC, COCO, Ablation, TPR@95% threshold)
  script/                    # Training entry (TrainAll.py)
  datasets/, checkpoint/, save/  # Data, pretrained weights, outputs (mounted as needed)

HDProof/                     # Hamming distance ZK proof (Rust/arkworks)
  3rd/                       # poly-commit, ml_sumcheck submodules
  src/                       # Protocol implementation
  src/bin/quick_bench.rs     # Benchmark entry (see CLI options below)

Dockerfile                   # Reproducible env (CUDA 12.1 + Python 3.10 + PyTorch 2.5.1 + Rust)
```


## Environment and Dependencies
We recommend Docker for consistent CUDA/driver and Python/Rust dependencies.

### Docker (recommended)
Build the image from the repo root:
```bash
git clone https://github.com/mengdehong/zkph.git
cd zkph
docker build -t zkph:latest .
```

Key environment in the image:
- Base: nvidia/cuda:12.1.1-devel-ubuntu20.04
- Python: 3.10 (Conda env: ph)
- PyTorch: 2.5.1 + CUDA 12.1 (torchvision 0.20.1, torchaudio 2.5.1)
- Other Python deps: see `PerceptHash/requirements.txt`
- Rust toolchain (stable) + arkworks
- `HDProof` is built during image build (`cargo build --release`).

Run-time recommendations:
- Use `--gpus all` for GPU
- Add `--shm-size=32g` to avoid DataLoader shared memory issues
- Mount host data into `/app/PerceptHash/...` inside the container via `-v`.

### Local Installation (optional)
If you do not use Docker, ensure compatible versions:
- CUDA 12.1 with PyTorch 2.5.1; Python 3.10
- Install deps from `PerceptHash/requirements.txt` (contains extra index for CUDA wheels)
- Rust toolchain (stable); build `HDProof` with `cargo build --release`

Note: CUDA/driver mismatches across hosts can break wheels; Docker is preferred for reproducibility.


## Data Preparation
Prepare a host data directory, e.g., `~/percepthash_data` (or reuse the original `~/code/zkph/PerceptHash/...` layout) and place:
- `PIHD` dataset (train & eval): expected to contain `PerceptHash/datasets/PIHD/test/test_class`
- `CocoVal2017` (eval/generalization): with `origin/`, `transformed/`, and `pairs_csv/`
- Pretrained weights (for training): put under `PerceptHash/checkpoint/`
- Trained models/logs (for evaluation): put under `PerceptHash/save/`

Mount these directories to `/app/PerceptHash/...` paths inside the container (see commands below).

### Dataset Links (from the original README)
To avoid link rot, both primary and backup links are listed:

1) PIHD dataset (train & eval)
   - Baidu Netdisk: https://pan.baidu.com/share/init?surl=uVnUVr5HqaSpoNifGElucw&pwd=8xwr
   - Google Drive (backup): https://drive.google.com/drive/folders/1gk9F8jv0Y4bX4JH3jv5y7K5y7K5y7K5y?usp=sharing

2) CocoVal2017 generated dataset (eval)
   - Google Drive: https://drive.google.com/file/d/10W5tuI6ZC-l4I_NkQgjmAHF5Gj0jx9Qw/view?usp=drive_link

3) Pretrained model weights (training)
   - Google Drive: https://drive.google.com/file/d/186XZ0lxF-rkDGFfLtaVrOJdpzog2hB_7/view?usp=drive_link

4) Trained outputs for evaluation
   - Google Drive: https://drive.google.com/file/d/1gk9F8jv0Y4bX4JH3jv5y7K5y7K5y7K5y/view?usp=drive_link


## Training and Evaluation
All commands are executed on the host; the container provides the runtime. Replace host paths as needed (e.g., `/home/USER/code/zkPH` or `~/percepthash_data`).

### Training (TrainAll)
Runs all configured backbones/strategies and saves the best weights under `save/TrainAll/<Model>/...` along with summaries:
```bash
docker run --gpus all -it --rm \
  --shm-size="32g" \
  -v ~/percepthash_data/PIHD:/app/PerceptHash/datasets/PIHD \
  -v ~/percepthash_data/checkpoint:/app/PerceptHash/checkpoint \
  -v ~/percepthash_data/save:/app/PerceptHash/save \
  zkph:latest \
  bash -c "source /opt/conda/bin/activate ph && cd PerceptHash && python script/TrainAll.py"
```

Global hyper-parameters and experiments are defined in `PerceptHash/script/TrainAll.py`.
Resource note: default `batch_size=64`; if memory is tight, reduce it accordingly.

### Evaluation: PIHD (AUC, FPR@95TPR)
`PerceptHash/eval/EvalAuc.py` supports three modes:
- 64-bit models: `--mode 64`
- 32-bit models: `--mode 32`
- TPR@95% Hamming threshold table for enhanced models: `--mode thr`

Optional args: `--batch-size`, `--num-workers`, `--device` (e.g., `cuda:0`/`cpu`), `--no-amp`.

Example:
```bash
docker run --gpus all -it --rm \
  --shm-size="32g" \
  -v ~/percepthash_data/PIHD:/app/PerceptHash/datasets/PIHD \
  -v ~/percepthash_data/save:/app/PerceptHash/save \
  -v ~/percepthash_data/checkpoint:/app/PerceptHash/checkpoint \
  zkph:latest \
  bash -c "source /opt/conda/bin/activate ph && cd PerceptHash && python -m eval.EvalAuc --mode 64 --batch-size 32 --num-workers 4"
```

Outputs CSV files under `save/eval/` with AUC and FPR@95TPR, and prints a summary.

### Evaluation: Params & Latency
Use `PerceptHash/eval/EvalParamLatency.py` to measure parameters and inference latency:
```bash
docker run --gpus all -it --rm \
  --shm-size="32g" \
  -v ~/percepthash_data/PIHD:/app/PerceptHash/datasets/PIHD \
  -v ~/percepthash_data/save:/app/PerceptHash/save \
  -v ~/percepthash_data/checkpoint:/app/PerceptHash/checkpoint \
  zkph:latest \
  bash -c "source /opt/conda/bin/activate ph && cd PerceptHash && python -m eval.EvalParamLatency"
```

### Evaluation: COCO Zero-shot Generalization
```bash
docker run --gpus all -it --rm \
  --shm-size="32g" \
  -v ~/percepthash_data/CocoVal2017:/app/PerceptHash/datasets/CocoVal2017 \
  -v ~/percepthash_data/save:/app/PerceptHash/save \
  -v ~/percepthash_data/checkpoint:/app/PerceptHash/checkpoint \
  zkph:latest \
  bash -c "source /opt/conda/bin/activate ph && cd PerceptHash && python -m eval.EvalCoco"
```

### Evaluation: Ablation Study
```bash
docker run --gpus all -it --rm \
  --shm-size="32g" \
  -v ~/percepthash_data/PIHD:/app/PerceptHash/datasets/PIHD \
  -v ~/percepthash_data/save:/app/PerceptHash/save \
  -v ~/percepthash_data/checkpoint:/app/PerceptHash/checkpoint \
  zkph:latest \
  bash -c "source /opt/conda/bin/activate ph && cd PerceptHash && python -m eval.EvalAblations"
```


## ZK Proof Benchmark (HDProof)
`HDProof` provides an arkworks-based ZK protocol for Hamming distance with a configurable benchmark `quick_bench` (built in Docker).

Example:
```bash
docker run -it --rm zkph:latest \
  HDProof/target/release/quick_bench \
  --sizes=14,16,18,20 --warmup=10 --samples=20 --seed=42 --tmpfs
```

Key options (from `src/bin/quick_bench.rs`):
- `--sizes=14,16,...`: number of hashes per run as 2^p; default 14,16,18
- `--warmup=N`: warm-up runs (default 1)
- `--samples=N`: number of samples; if omitted, `--repeats` (default 3) is used
- `--seed=S`: RNG seed for reproducibility
- `--tmpfs`: write intermediate files under `/dev/shm` if available to reduce I/O jitter


## Reproducibility Checklist and Expected Artifacts
1) Environment: Docker image above (CUDA 12.1 / PyTorch 2.5.1 / Python 3.10 / Rust stable)
2) Data:
   - PIHD: ensure `PerceptHash/datasets/PIHD/test/test_class` exists
   - COCOVal2017: ensure `origin/`, `transformed/`, `pairs_csv/`
   - `checkpoint/` and `save/` as needed
3) Training: run TrainAll -> `save/TrainAll/<Model>/..._best.pth`, `aggregated_results.json`
4) PIHD eval: `EvalAuc --mode 64/32/thr`
5) COCO eval: `EvalCoco`
6) Ablation: `EvalAblations`
7) ZK benchmark: `quick_bench` -> collect statistics (mean/median/stddev/p90/min/max)


## FAQ
- CUDA/driver mismatch: Prefer Docker; for local installs ensure CUDA and PyTorch versions match
- DataLoader shared memory: add `--shm-size=32g` or reduce `--num-workers`
- Out-of-memory: reduce `--batch-size` or use CPU (`--device cpu`) at the cost of speed
- Missing weights for eval: ensure `save/TrainAll/..._best.pth` is mounted inside the container
- COCO eval requires three directories: `origin/`, `transformed/`, `pairs_csv/`
- I/O jitter in `quick_bench`: add `--tmpfs`


## Acknowledgements
We gratefully acknowledge the following open-source projects and repositories that inspired or supported parts of this work:
- arkworks: https://github.com/arkworks-rs
- GroupMamba: https://github.com/Amshaker/GroupMamba
- MambaHash: https://github.com/shuaichaochao/MambaHash
- DinoHash (perceptual hash): https://github.com/proteus-photos/dinohash-perceptual-hash.git

Notes:
- We do not provide environments or integrations for Apple NeuralHash or imagededup. Experiments involving those ecosystems are out-of-scope of this repository’s environment setup.


License note: Components under `HDProof` (hamproof crate) rely on arkworks crates and follow their upstream licenses. If no unified license is specified at the repo root, please use this code for academic research purposes.
