# Experiments needed before claiming improvement

1. Obtain paired UIEB data and freeze the manifest; review scenes for near-duplicates. Publish filenames and hashes, not dataset images without permission.
2. Train B0 baseline, N2 NPS, R1 RA-Q/K and C2 RA+NPS under identical configs/seeds. Keep the 90 test images out of selection. Include spatial_control and unit_qk to separate phase-specific accuracy from generic pooling and numerical speed changes.
3. Select architecture/hyperparameters on validation only. Report 3-seed mean/std; pair images for confidence intervals and avoid treating 90 correlated image results as 90 independent training replicates.
4. Repeat the strongest comparison with the fixed four-loss objective. If the improvement disappears, report that dependency.
5. Publish baseline and proposed checkpoints, actual logs, exact data manifest and per-image metrics. Remove 'pending' only when files exist and checksums are recorded.
6. Run same-resolution, same-device cost profiling. Keep MAC/FLOP conventions and FFT omissions visible. Test CUDA instead of extrapolating CPU speed.
7. Inspect input/baseline/proposed/reference image grids, including failures. Evaluate noise robustness and a separately specified blur perturbation; examine edge loss and colour overshoot.
8. Evaluate a verified external dataset under a clearly defined native split. Do not apply unknown-split UIEB weights to new UIEB test images as if they were guaranteed unseen.

A negative result is acceptable research evidence. If NPS does not outperform the matched controls, describe it as a diagnosed architectural limitation with an unsuccessful repair rather than relabeling a training-pipeline change as architectural novelty.
