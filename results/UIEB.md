# UIEB experiment artifacts

## Held-out comparison

| Metric | Phaseformer | NPS-Phaseformer | NPS − baseline | Paired 95% CI |
|---|---:|---:|---:|---|
| psnr_rgb | 24.23714 | 24.31619 | +0.07905 | [-0.282516, 0.440046] |
| ssim_rgb | 0.904149 | 0.908587 | +0.004438 | [0.000455, 0.008880] |
| uiqm_compat | 3.193273 | 3.165867 | -0.027406 | [-0.048774, -0.006383] |

NPS improved SSIM and reduced compatibility UIQM. Mean PSNR was slightly higher, but its paired interval includes zero. Intervals resample test images for this trained pair and do not measure variability across training seeds.

## Run records and reproduction

[Full metric and latency table](ui/report_table.csv) · [Paired bootstrap output](ui/comparison.json) · [T4 timings and FLOP convention](ui/benchmark.json)

Both models completed 30 epochs (21,600 updates each) on a 200-epoch cosine schedule; validation PSNR peaked at epoch 27 for [baseline](ui/baseline/train.jsonl) and 26 for [NPS](ui/nps/train.jsonl). [Baseline settings](ui/baseline/config.json) and [NPS settings](ui/nps/config.json) record Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 and Tesla T4. The split comprises 720 training, 80 validation and 90 test pairs. The export contains aggregate scores but not the original per-image CSVs or test summaries. Selected checkpoint hashes are available [here](ui/checkpoint_sha256.json).

[Exact split manifest and source-image checksums](ui/uieb.json) · [Training and validation curves](ui/learning_curves.png). When evaluating locally, set the manifest’s `root` to your UIEB directory; keep the sample lists and file hashes unchanged.
