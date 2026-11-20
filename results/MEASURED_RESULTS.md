# Measured execution evidence

CPU only, FP32, 256×256 input, batch 1, both heads computed, 1 PyTorch thread; 2 warmup and 5 measured calls per variant. Random weights; these are execution measurements, not accuracy results.

| Variant | Registered parameters | Executed parameters | Conv+matmul GFLOPs (partial) | FFT estimate (GFLOPs) | Median CPU ms |
|---|---:|---:|---:|---:|---:|
| baseline | 1,783,800 | 1,779,189 | 27.839300 | 6.3334 | 4147.2 |
| unit_qk | 1,783,800 | 1,779,189 | 27.839300 | 6.3334 | 2727.1 |
| nps | 1,783,818 | 1,779,207 | 27.839302 | 6.3334 | 4255.0 |
| ra_qk | 1,783,848 | 1,779,237 | 27.839300 | 6.3334 | 2478.2 |
| ra_nps | 1,783,866 | 1,779,255 | 27.839302 | 6.3334 | 2496.1 |

NPS adds 18 parameters (0.00101%). Its median CPU latency was 2.60% higher in this small run. This is indicative, not a statistically established device-independent overhead. Raw samples are retained in the JSON files.

The unit_qk algebra-only control also runs faster than the baseline: part of the apparent RA speedup comes from replacing angle/complex-exp, not from learning reliability. No GPU timing was run.

FLOP columns exclude pooling, nonlinearities, normalization and elementwise gates. Conv/matmul uses two operations per MAC; FFT/IFFT is an analytical estimate. Do not call either column, or their sum, exact total FLOPs. The paper reports 13.0 GFLOPs under a different/insufficiently specified counting setup. The measured registered count of 1,783,800 is also not numerically identical to the paper's 1.77M.

## Accuracy status

UIEB/UFO-120 proposed-model PSNR, SSIM and UIQM are **not measured**. Published scores belong only to the original paper; synthetic smoke metrics establish software execution only. See `experiment_status.csv` and the research report.
