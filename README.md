# NPS-Phaseformer: a controlled study of phase-statistic skip attention

A complete research extension of **Phaseformer: Phase-Based Attention Mechanism for Underwater Image Restoration and Beyond (WACV 2025)**, with its original backbone, official UIEB weights, a minimal proposed module, RA-Phaseformer controls, training/evaluation code, tests and a critical research report.

**Start here:** [Kaggle training notebook](notebooks/Phaseformer_Kaggle.ipynb) · [Architecture and derivation](docs/RESEARCH_REPORT.md) · [Module map](docs/CODE_MAP.md) · [Experiment artifacts](results/UIEB.md).

The UIEB experiment runs one model per NVIDIA Tesla T4. The same paired split, initialization seed and training recipe apply to both models. The team’s Kaggle evaluation on the 90-image UIEB test split is summarized below.

| Metric | Phaseformer | NPS-Phaseformer | NPS − baseline | Paired 95% CI |
|---|---:|---:|---:|---|
| psnr_rgb | 24.23714 | 24.31619 | +0.07905 | [-0.282516, 0.440046] |
| ssim_rgb | 0.904149 | 0.908587 | +0.004438 | [0.000455, 0.008880] |
| uiqm_compat | 3.193273 | 3.165867 | -0.027406 | [-0.048774, -0.006383] |

NPS improved SSIM and reduced compatibility UIQM. Mean PSNR was slightly higher, but its paired interval includes zero. Intervals resample test images for this trained pair and do not measure variability across training seeds.

**Final run:** Both models completed 30 epochs. Validation selected baseline epoch 27 and NPS epoch 26. On a Tesla T4 the median latency increased from 105.18 to 108.45 ms (+3.1%) for 18 additional parameters. [Raw training logs, performance table and split](results/ui/) · [Checkpoint instructions](results/UIEB.md).

## 1. Project overview and precise contribution

The released skip attention computes `GAP(IFFT(exp(i*phase)))`. Under the default FFT normalization this is only the DC phasor divided by the pixel count. Non-DC phase structure is lost. Simply adding amplitude reliability weights does not repair that pooling limitation.

**NPS** retains the original skip logit and adds a zero-initialized channel convolution over two nonlinear statistics of the centred phase reconstruction: normalized mean absolute response and normalized mean absolute 3×3 local-average response. It introduces **18 parameters**, needs no additional FFT, and initially gives exactly the baseline's predictions. Better restoration remains an experimental hypothesis.

Included variants:

| `--variant` | Change from released backbone |
|---|---|
| `baseline` | None |
| `unit_qk` | Algebra-only `F/abs(F)` phase extraction, no reliability weighting; latency control |
| `nps` | Proposed two-statistic phase skip correction |
| `nps_l1` | Single-statistic ablation |
| `spatial_control` | Same-size correction using spatial feature statistics |
| `ra_qk` | Reliability-weighted PEM for Q/K only |
| `ra_skip` | Reliability-weighted PEM for skips with original GAP |
| `ra_both` | Reliability-weighted PEM in Q/K and skips |
| `ra_nps` | RA Q/K plus NPS skips; skip PEM stays original |

The `unit_qk` control separates savings from avoiding complex `angle/exp` from the reliability mechanism itself.

RA uses `R=A²/(A²+tau²)` with scale-normalized positive tau, implemented without explicit phase angles. `--fixed-tau` and `--scale ac` expose threshold ablations. The report explains why observed amplitude is only a reliability proxy and why broad spectral-attention novelty claims are unjustified.

References: [paper](https://openaccess.thecvf.com/content/WACV2025/html/Khan_Phaseformer_Phase-Based_Attention_Mechanism_for_Underwater_Image_Restoration_and_Beyond_WACV_2025_paper.html), [official implementation](https://github.com/Mdraqibkhan/Phaseformer), [FUSION](https://github.com/shravan-18/FUSION), [Spectroformer](https://github.com/Mdraqibkhan/Spectroformer). Original backbone copyright and MIT license are retained. Related source snapshots and findings are listed in `docs/source_manifest.json` and the report.

## 2. Environment and dependencies

Clone the repository and run the commands from its root:

```bash
git clone https://github.com/seedhecode75/NPS-Phaseformer.git
cd NPS-Phaseformer
```

 **Python 3.12** is the supported project target. The actual executed environment was Python 3.12.14, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Linux x86_64, single PyTorch CPU thread, no CUDA device. Complete package versions are in `requirements-tested.txt`; hardware details are in `results/environment.json`.

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1` (use `py -3.12` in place of `python3.12`).

**Exact tested CPU stack:**

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q tests
```

**Alternative pinned CUDA target, not executed in this environment:**

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pytest -q tests
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

This is the PyTorch project's documented 2.5.1 / torchvision 0.20.1 / CUDA 12.1 wheel pairing, not a claim of GPU validation. See [official wheel instructions](https://pytorch.org/get-started/previous-versions/). Install a compatible NVIDIA driver. No custom CUDA extension or local CUDA compiler is required. Use an NVIDIA GPU for practical full training; start at batch 1 and measure memory on your device. The paper used an RTX 2080; this project does not claim its memory usage or runtime was reproduced. The trainer uses FP32; only the standalone reliability extractor was BF16-tested on CPU.

## 3. Dataset preparation

Download data yourself from the original providers:

- **UIEB:** [official benchmark page](https://li-chongyi.github.io/proj_benchmark.html), including 890 raw/reference pairs. Follow its noncommercial research terms. The extra 60 challenging images are unpaired and must not enter paired metrics.
- **UFO-120:** [official dataset page](https://irvlab.cs.umn.edu/resources/ufo-120-dataset). Keep its provided held-out test set; do not combine it with training data before splitting. Align the correct enhancement/SR target convention first.

Primary executable protocol in this package is UIEB. Unpack into:

```text
datasets/UIEB/
  raw-890/        image1.png, image2.png, ...
  reference-890/  image1.png, image2.png, ...
```

Filenames are matched by stem; extensions may differ. Duplicate stems, missing pairs and exact file duplication across splits are rejected. Prepare once:

```bash
python scripts/prepare_data.py --root datasets/UIEB --out splits/uieb.json --seed 2025 --test-count 90 --val-count 80
```

This makes **720 training / 80 validation / 90 test** pairs. It is a new controlled split, **not** a reproduction of the unknown official Phaseformer split. Preserve `splits/uieb.json` with the results; it records filenames, file hashes and the absolute local root. To relocate an existing manifest without changing its membership, edit only its `root` value and record the resulting manifest hash. Do not regenerate a split to obtain better scores.

If a verified benchmark test list is available, use `--test-list path/to/test.txt`; each line contains a filename or stem. The released Phaseformer repository does not supply its exact test list. Do not assume FGDPA's list matches it.

Preprocessing is PIL bicubic resize to 256×256 input/low target and 512×512 high target; convert to RGB; input normalized to [-1,1], targets retained in [0,1]. Training applies aligned random horizontal/vertical flips. No noise or contrast augmentation is silently added. File hashes are rechecked at training and paired evaluation startup. Hash checks do not replace a scene-level near-duplicate audit.

### UFO-120 with its native held-out set

Organize the chosen enhancement input and paired target files as `datasets/UFO120/train/input`, `train/target`, `test/input`, `test/target`, preserving the provider's train/test membership. Choose the enhancement task rather than silently mixing different SR scales. Filenames must match by stem within each split.

```bash
python scripts/prepare_native.py --root datasets/UFO120 --val-count 150 --out splits/ufo120.json
python train.py --manifest splits/ufo120.json --variant baseline --device cuda --output runs/ufo_baseline
python train.py --manifest splits/ufo120.json --variant nps --device cuda --output runs/ufo_nps
python evaluate.py --manifest splits/ufo120.json --checkpoint runs/ufo_baseline/best.pt --device cuda --output runs/ufo_baseline/test
python evaluate.py --manifest splits/ufo120.json --checkpoint runs/ufo_nps/best.pt --device cuda --output runs/ufo_nps/test
```

With 1,500 provided training pairs this reserves 150 for validation and trains on 1,350; all 120 native test pairs remain held out. This validation holdout and our declared resize/head protocol mean these runs are not an exact reproduction of the paper's 1,500-pair training result. No UFO-120 run was executed here.

## 4. Training and ablations

The new trainer fixes execution/range problems equally for every model. The original scripts remain in `legacy/` for audit; do not use `legacy/train.py` as a working reproduction recipe. Its bugs are documented in the report.

**Primary paired runs, from scratch:** For two T4s, launch these commands concurrently in separate terminals, replacing the first device with `cuda:0` and the second with `cuda:1`. Both also work sequentially on one GPU.

```bash
python train.py --config configs/controlled.json --manifest splits/uieb.json --variant baseline --seed 123 --device cuda --output runs/baseline_seed123
python train.py --config configs/controlled.json --manifest splits/uieb.json --variant nps --seed 123 --device cuda --output runs/nps_seed123
```

`controlled.json`: 200 epochs, batch 1, Adam 3e-4, betas (.5,.999), cosine schedule, gradient clipping 1.0, 256px input, `.6*L_low + .4*L_high`, Charbonnier epsilon .001. Both heads are supervised. This is a chosen controlled recipe; it is not a claim to recover the paper's original training settings.

**Full core study with three seeds:**

```bash
python scripts/run_ablation.py --manifest splits/uieb.json --device cuda --seeds 123 456 789 --variants baseline nps ra_qk ra_nps spatial_control
```

Other controls:

```bash
python train.py --manifest splits/uieb.json --variant ra_qk --fixed-tau --seed 123 --device cuda --output runs/ra_fixed_seed123
python train.py --manifest splits/uieb.json --variant ra_qk --scale ac --seed 123 --device cuda --output runs/ra_ac_seed123
python train.py --manifest splits/uieb.json --variant ra_skip --seed 123 --device cuda --output runs/ra_skip_seed123
python train.py --manifest splits/uieb.json --variant ra_both --seed 123 --device cuda --output runs/ra_both_seed123
python train.py --manifest splits/uieb.json --variant nps_l1 --seed 123 --device cuda --output runs/nps_l1_seed123
```

Resume at the last saved **epoch boundary**, retaining the same config and total epoch budget:

```bash
python train.py --config configs/controlled.json --manifest splits/uieb.json --variant nps --seed 123 --device cuda --output runs/nps_seed123 --resume runs/nps_seed123/last.pt
```

Checkpoints preserve model, optimizer, scheduler, Python/PyTorch RNG states, model options, seed, manifest hash and environment. Data loading uses zero workers for straightforward deterministic replay. CPU↔GPU and different-library-version replay are not promised to be bitwise identical.

**Secondary multi-loss protocol:** use `--config configs/paper_fixed.json` for *both* baseline and proposed runs. It uses actual Charbonnier, Laplacian gradient L1, `1-MS_SSIM` and VGG16 L1 with fixed weights (.2741,.2222,.3357,.1680) in that order. Torchvision downloads ImageNet VGG16 weights on first use. Inputs to perceptual/SSIM terms use the correct intensity convention. This path is implemented; full training was not run here. It is explicitly fixed-weight and does not claim to reproduce the paper's dynamic-loss process. Its minimum training size is >160 due to five-level MS-SSIM.

**Pretrained initialization:** `--init checkpoints/upstream_uieb_state.pt` works for baseline and all extensions. Use only for a verified disjoint fine-tuning experiment and give baseline the same initialization/budget. The official checkpoint's unknown training split makes it unsuitable for a leakage-free comparison on the new UIEB split.

## 5. Inference and evaluation

**Real-image inference using the original authors’ pretrained weights:**

```bash
python scripts/fetch_upstream.py
python evaluate.py --checkpoint checkpoints/upstream_uieb_state.pt --inputs path/to/underwater_images --size 256 --head low --device cpu --output outputs/upstream_demo
```

Images are read from the specified folder. Original aspect ratio is deliberately resized to a square for this declared protocol. Use `--device cuda` for GPU. `--size` must be divisible by 8. Outputs are RGB PNGs. Using `--head high` returns twice the specified input size. The upstream demo used `--size 512 --head high`; that is a different 1024px output protocol and is much more expensive.

**Evaluate trained models on the frozen test set:**

```bash
python evaluate.py --manifest splits/uieb.json --checkpoint runs/baseline_seed123/best.pt --device cuda --output runs/baseline_seed123/test
python evaluate.py --manifest splits/uieb.json --checkpoint runs/nps_seed123/best.pt --device cuda --output runs/nps_seed123/test
python scripts/compare_results.py --baseline runs/baseline_seed123/test --proposed runs/nps_seed123/test --out results/uieb_seed123_comparison.json
```

The model variant is recovered from the checkpoint. `best.pt` is selected only by validation low-head float RGB PSNR. Evaluation defaults to 256px low-head output. It writes `summary.json`, `per_image.csv` and output PNGs. Metrics use clipped/rounded uint8 RGB, per-image PSNR and Gaussian SSIM with sigma 1.5, no border cropping. `uiqm_compat` is a documented FUSION-derived UIQM variant; do not equate it automatically with an unknown paper implementation.

**Noise stress test:** both methods must use the same noise/seed settings:

```bash
python evaluate.py --manifest splits/uieb.json --checkpoint runs/baseline_seed123/best.pt --noise 0.0392156862745098 --seed 123 --device cuda --output runs/baseline_seed123/noise10
python evaluate.py --manifest splits/uieb.json --checkpoint runs/nps_seed123/best.pt --noise 0.0392156862745098 --seed 123 --device cuda --output runs/nps_seed123/noise10
```

**Compute cost and inference speed:**

```bash
python benchmark.py --device cpu --size 256 --threads 1 --warmup 5 --runs 20 --out results/benchmark_local_cpu.json
python benchmark.py --device cuda --size 256 --warmup 20 --runs 100 --out results/benchmark_local_gpu.json
```

Measures both output heads, registered/executed parameters, synchronized latency and GPU peak allocated memory. Conv and attention matmul counts use 2 FLOPs/MAC. FFT work is reported separately as an estimate; nonlinearities, normalization, pooling and elementwise work are excluded. **These are not full-network exact FLOPs and cannot be directly compared with the paper's 13.0 GFLOPs.** CPU results cannot be compared with an RTX GPU runtime.

## 6. Checkpoints, logs and result status

| Item | Location | Status |
|---|---|---|
| Official UIEB state dictionary | [Download and conversion script](scripts/fetch_upstream.py) | Rebuild locally from the checksum-verified original |
| Official source checkpoint | [pinned upstream UIEB weights](https://github.com/Mdraqibkhan/Phaseformer/blob/6e759dc782e514c985672c545205dea86a3d21b5/checkpoints/UIEB/best.pth) | Public source and SHA recorded |
| Conversion provenance | [checkpoints/upstream_uieb_state.json](checkpoints/upstream_uieb_state.json) | Input SHA-256 and source commit |
| Proposed underwater weights | [Artifact index](results/UIEB.md) | Checkpoint publication pending |
| Trained baseline/proposed results | Generated under `runs/` by commands above | [Reported comparison](results/UIEB.md); per-image CSVs can be regenerated |
| Test log | [results/tests.xml](results/tests.xml) | Executed unit/numerical checks |
| Mechanism evidence | [results/mechanism_probe.json](results/mechanism_probe.json) | Executed synthetic diagnostics |
| CPU profile | [results/benchmark_cpu.json](results/benchmark_cpu.json) | Executed timing/parameter counts |
| Smoke logs | [results/smoke/](results/smoke/) | One-step random-image software tests; not research scores |

To recreate conversion after downloading the original pickle:

```bash
python scripts/convert_checkpoint.py --input path/to/original/best.pth --output checkpoints/upstream_uieb_state.pt
```

Conversion uses a restricted allowlist of the inspected architecture classes, then strictly checks state shapes against the source constructor. Routine loading uses state dictionaries with `weights_only=True`.

See [UIEB experiment artifacts](results/UIEB.md) for the release contents and evaluation protocol.

## 7. Code organization and reproduction checks

```text
model_with_eca.py                 released architecture; only two unused imports removed
phaseformer_ext/modules.py       proposed skip statistics and reliability extractor
phaseformer_ext/model.py         explicit variant factory; strict baseline-weight transfer
phaseformer_ext/data.py          paired data and file/split integrity
phaseformer_ext/losses.py        shared controlled and fixed-four-loss objectives
phaseformer_ext/metrics.py       RGB PSNR/SSIM and explicit quantization
phaseformer_ext/uiqm_compat.py   attributed compatibility UIQM implementation
train.py                        deterministic shared trainer and epoch resume
evaluate.py                     inference, paired/unpaired evaluation, noise probes
benchmark.py                    honest partial FLOP accounting and synchronized timing
scripts/prepare_data.py          immutable split manifest
scripts/run_ablation.py          repeatable variant × seed runs
scripts/compare_results.py       paired image bootstrap
scripts/mechanism_probe.py       DC-collapse and pretrained-initialization evidence
tests/test_research.py           numerical and full-model gradient checks
legacy/                         untouched upstream Python scripts for audit
docs/RESEARCH_REPORT.md          four assignment sections and literature/code critique
```

[Code map](docs/CODE_MAP.md) gives the precise change sites. [Patch](docs/extension.patch) shows the implementation relative to a source checkout. The original architecture, license and acknowledgements stay visible.

**Fast pipeline check without a dataset or GPU:** `python scripts/smoke.py` uses a fresh temporary directory automatically. The individual commands are:

```bash
python -m pytest -q tests
python scripts/mechanism_probe.py
python scripts/make_smoke_data.py --root tmp/smoke_data
python scripts/prepare_data.py --root tmp/smoke_data --test-count 2 --val-count 2 --out tmp/smoke.json
python train.py --manifest tmp/smoke.json --variant nps --device cpu --size 32 --epochs 1 --max-steps 1 --output tmp/nps_smoke
python evaluate.py --manifest tmp/smoke.json --checkpoint tmp/nps_smoke/best.pt --size 32 --device cpu --output tmp/smoke_eval
```

Use fresh `tmp` paths when repeating; the split builder intentionally refuses overwrite. `--max-steps` marks checkpoints as smoke-only and is not part of full training.

