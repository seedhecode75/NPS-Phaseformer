# UIEB experiment artifacts

## Held-out comparison

| Metric | Phaseformer | NPS-Phaseformer | NPS − baseline | Paired 95% CI |
|---|---:|---:|---:|---|
| psnr_rgb | 24.23714 | 24.31619 | +0.07905 | [-0.282516, 0.440046] |
| ssim_rgb | 0.904149 | 0.908587 | +0.004438 | [0.000455, 0.008880] |
| uiqm_compat | 3.193273 | 3.165867 | -0.027406 | [-0.048774, -0.006383] |

NPS improved SSIM and reduced compatibility UIQM. Mean PSNR was slightly higher, but its paired interval includes zero. Intervals resample test images for this trained pair and do not measure variability across training seeds.

Source: aggregate output supplied from the team's Kaggle comparison. [Download as CSV](uieb_comparison.csv). Checkpoints, logs, and raw per-image files have not yet been published. The original authors’ [pretrained UIEB checkpoint](https://github.com/Mdraqibkhan/Phaseformer/blob/6e759dc782e514c985672c545205dea86a3d21b5/checkpoints/UIEB/best.pth) is available separately.

**Protocol:** 720 training / 80 validation / 90 test images, split seed 2025, model seed 123. Both architectures start from scratch. Select by validation RGB PSNR within the same completed training horizon. Record the actual completed and selected epochs from the logs alongside the [scheduler configuration](../configs/controlled.json).

**Release contents:** selected weights and SHA-256 for each model, per-run config and training log, split manifest, environment, both test summaries and per-image CSVs, paired comparison, GPU benchmark and example panel. Checkpoint hashes must match the evaluation summaries. Report PSNR, SSIM, compatibility UIQM and image-bootstrap intervals; label the separate FLOP estimates and the GPU used for latency.

Use the commands in [Inference and evaluation](../README.md#5-inference-and-evaluation) to regenerate the tables from the released weights. Software smoke tests under this directory are separate from the held-out experiment.
