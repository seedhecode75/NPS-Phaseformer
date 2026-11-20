# Validation boundaries

Executed: mathematical/numerical tests; full forward/backward for all model variants on CPU; safe conversion and state loading of the original UIEB checkpoint; exact zero-initialization parity; synthetic one-step training, save/reload, PNG inference, PSNR/SSIM/UIQM and paired comparison; CPU parameter/timing/partial-operation measurements.

Not executed: full underwater dataset training, the VGG-based four-loss training run, CUDA training/inference, complete original-paper score reproduction, cross-dataset generalization or a publication-level exhaustive novelty search.

No underwater-trained proposed weights are included. `results/smoke/` is synthetic debugging evidence. Do not paste its numbers into a UIEB/UFO table. Measured CPU profiling excludes data decoding, output saving and metrics.

The data and training recipes are reproducible new controlled protocols, not recovered versions of the authors' unreleased test split and training history.
