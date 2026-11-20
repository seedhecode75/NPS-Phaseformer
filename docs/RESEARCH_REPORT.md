# Phaseformer research audit and proposed extension

**Working project:** NPS-Phaseformer — Nonlinear Phase-Statistic Skip Attention  
**Assessment date:** 17 October 2026  
**Evidence status:** source-audited, implemented, CPU-tested; underwater training and superiority unverified.

## Decision before implementation

Do not present the handoff's RA-Phaseformer as an established improvement or a first-ever reliability-weighted spectral method. Its mechanism is plausible for low-SNR coefficients, but its reliability interpretation is heuristic. More importantly, applying it only to Phaseformer's skip attention leaves a structural information bottleneck intact.

**Recommended primary experiment: NPS-Phaseformer.** Preserve the released backbone and add an 18-parameter, zero-initialized nonlinear phase-statistic branch to its three active skip gates. Run RA query/key weighting as an independent factor; `ra_nps` combines these changes. Select a winner only on validation data. There is no evidence yet that the combination is better than the smaller repair.

The strongest current contribution is a specific, testable diagnosis and a minimal correction, not a demonstrated state-of-the-art result. The project is suitable for an honest research assignment once controlled training is completed. Publication-level originality and performance cannot be certified by this audit alone.

## 1. Summary of the Assigned Paper

### Problem and motivation

Underwater absorption and scattering change colour, suppress contrast, and obscure structure. The wavelength dependence matters: a single colour correction cannot generally undo every location's degradation. A common explanatory image model is

\[
I_c(x)=J_c(x)t_c(x)+B_c(x)(1-t_c(x)),\qquad t_c(x)=e^{-\beta_c d(x)}.
\]

Phaseformer does not explicitly solve this physical inverse problem. It instead tries to choose better features for computing attention. In a Fourier representation

\[
F_X(\omega)=A_X(\omega)e^{i\phi_X(\omega)},
\]

magnitude describes spectral energy while relative phase strongly influences spatial arrangement. The paper's t-SNE comparison motivates using phase for attention because its clean/degraded clusters overlap more. This is a qualitative observation, not proof that phase is unaffected by noise or that underwater degradation changes only magnitude.

The network uses a U-shaped Restormer-style hierarchy, phase-based channel attention, skip gates and two output resolutions. Queries and keys use phase-only features; values retain conventional projected features. Therefore, amplitude information is not removed from the whole restoration network. The output can still recover colour using the value and convolution paths.

### Equations versus actual execution

The phase extractor is

\[
P(X)=\operatorname{Re}\{\mathcal F^{-1}[e^{i\arg\mathcal F(X)}]\}.
\]

The released `model_with_eca.py::MDTA.forward` first computes pointwise and depthwise projections, then applies `inv_mag` separately to Q and K. The paper's Eq. (1) and Figure 2 instead place PEM before these projections. The provided project reproduces the **released code order** for checkpoint compatibility; it does not silently substitute the schematic.

For B examples, h heads, d=C/h channels/head and N=HW pixels, Q,K,V have shape B×h×d×N. The code normalizes Q,K along N and computes

\[
A=\operatorname{softmax}(\hat Q\hat K^T\odot T),\qquad O=W_o(AV),
\]

where A is B×h×d×d and T is a learned per-head multiplier. It is channel-transposed attention, not an N×N spatial attention map. Calling it ordinary pixel-to-pixel attention would obscure the actual complexity and mechanism.

The skip gate is `ECA.forward`:

\[
Z=X\odot\sigma\{\operatorname{Conv1D}[\operatorname{GAP}(P(X))]\}.
\]

Its kernel is chosen at construction from a channel-count formula; kernel size does not learn dynamically during training. The learned quantities are convolution weights. The two outputs have input resolution and twice the input resolution.

### What the paper reports

These are transcribed **published results**, not reproduced scores:

| Dataset / model | PSNR dB | SSIM | UIQM |
|---|---:|---:|---:|
| UIEB / Spectroformer | 24.96 | 0.917 | 3.075 |
| UIEB / Phaseformer | 25.98 | 0.928 | 4.451 |
| UFO-120 / Phaseformer | 30.38 | 0.878 | 2.907 |

The paper reports 1.77M parameters and 13.0 GFLOPs. Its U45 runtime table labels a column `Time(image/sec)` with a downward arrow, while the discussion describes seconds/image. Treat the reported 0.034 as unit-ambiguous rather than converting it into a confident FPS claim. The paper acknowledges difficulty with muddy/blurry images. UIEB is a real-image enhancement benchmark with selected reference enhancements; describing it uncritically as physically exact synthetic ground truth would be misleading.

## 2. Proposed Novelty and Technical Contribution

### Code audit: issues that change the experiment

| Exact released location | Verified observation | Experimental consequence |
|---|---|---|
| `model_with_eca.py::inv_mag`, lines 13–16 | FFT → unit phasor → real IFFT | Zero coefficients receive the library's angle-zero convention; all spectral magnitudes become one. |
| `ECA.forward`, lines 43–58 | Immediately averages phase-only reconstruction | Gate descriptor retains only DC, not the non-DC phase structure. |
| `MDTA.forward`, lines 72–87 | Projections precede PEM; Q/K spatial L2 normalization | Preserve this order; a channel-wide positive scaling of Q/K is cancelled by normalization. |
| `Restormer.__init__/forward` | Four ECA modules registered; only first three used; `ups2` also unused | Report registered and executed parameter counts; do not remove dead weights in only one model. |
| `network1.py` | GAN/ResNet helpers and scheduler, not Phaseformer backbone | Editing its generator would modify the wrong model. |
| `train.py::WeightedLoss`, lines 197–209; optimizer at 229 | Softmax makes weights positive/normalized, but optimizer contains only `net_g.parameters()` | They receive gradients but are not updated; this is not unconstrained negative weighting, as the handoff wondered. |
| `train.py`, lines 223, 242–244 | Adds MS-SSIM similarity with positive sign to a minimized objective | Use `1-MS_SSIM` in a clearly labeled repaired protocol for every model. |
| `train.py`, lines 184–194 and 224 | Defines Charbonnier but instantiates `SmoothL1Loss` | Paper's loss name and executed loss differ. |
| `train.py::get_dataset`; `utils.py::VGGPerceptualLoss` | Images normalized to [-1,1]; MS-SSIM uses range 1 and VGG assumes [0,1] ImageNet normalization | Correct ranges consistently in the new shared trainer. |
| `train.py`, line 273 | Uses undefined `avg_psnr` in checkpoint filename | Unmodified released training cannot complete checkpoint saving. |
| `train.py` argument handling | Dataset/batch paths partly hard-coded; CUDA default cannot be switched off normally; lambda scheduler default | Rebuild CLI and record all executed settings. |
| `dataset.py::DatasetFromFolder`; `test.py` | Inputs resized to 512; output `[1]` selected | Released inference produces 1024px output; do not compare against our 256px low-head protocol without qualification. |

Even adding the softmax loss weights to Adam would not by itself establish a sound adaptive balancing scheme: for L=sum_j w_j L_j and w=softmax(a), dL/da_j=w_j(L_j−L). Direct minimization encourages weight on the easiest/smallest-scale losses. A regularized or otherwise justified balancing algorithm would need its own ablation; this project uses fixed weights instead.

The loss-optimizer issue is also raised in upstream issue #7. Issue #6 requests the exact UIEB test filenames; no official list was found in the audited checkout. Consequently the released weights cannot establish leakage-free performance on an arbitrary new split.

### Prior-art comparison

Source commits and inspected Python hashes are retained in `source_manifest.json`. Related code was read, not executed as a benchmark.

| Method/source | Actual mechanism inspected | Implication for novelty |
|---|---|---|
| FUSION, `models/frequency.py::FrequencyBranchProcessor` | rFFT; learned magnitude transformations and channel attention; recombination with original phase | Broad amplitude-guided, phase-preserving frequency processing is already established. RA is narrower: bounded shrinkage of unit phasors used for Q/K. |
| FUSION, `FrequencyGuidedFusion.forward` | Instantiates a channel adapter inside `forward` when widths differ; current network passes 16-channel frequency features to a 1-channel fusion module | A freshly created, unregistered adapter is a reproducibility risk. Do not copy this code as a ready-made improvement. No claim here that its published results used this exact checkout. |
| Spectroformer, `model.py::{MDTA,SFCA,UpSample}` | Cascaded spatial/frequency attention; real FFT paths; learned magnitude and phase upsampling | A generic dual-domain branch is heavily overlapping and more complicated. `final_model_AGSSF1.py` is an alternative file with a phase-only pooled skip and differing forward code; do not conflate all files with a single definitive model. |
| UIEPTA, `modifiedmodel1.py::Phase_swap` | Combines one feature's phase with another's magnitude in skip transfers | Phase-guided restoration predates Phaseformer. Its `fftn` omits `dim`, transforming batch/channel axes too; not a drop-in template. |
| FGDPA, `model/uie.py::_fgdra_attention` | Downsampled FFT log-magnitude mean plus spatial features drive channel/spatial modulation | Frequency statistics for attention are prior art. It does not implement the tested unit-phase nonlinear statistics. |
| FcaNet, `model/layer.py::MultiSpectralAttentionLayer` | DCT projections at selected frequencies feed channel attention | GAP's low-frequency limitation is already recognized. We cannot claim discovering that GAP corresponds to DC as a general theorem. |
| Kovesi's phase congruency work | Phase alignment, amplitude weighting and noise compensation for feature significance | “Use phase confidence to reject noise” is not a new general idea. NPS is not a phase-congruency estimator. |

Five visible Phaseformer forks were compared through GitHub's compare API: all were behind the upstream main branch, with no ahead commits. This is a dated inspection of default branches, not proof that no improvement exists anywhere or on unexamined branches. See `fork_audit.json`.

### Critical verdict on RA-Phaseformer

The handoff proposes

\[
R(A)=\frac{A^2}{A^2+\tau^2},\qquad P_R(X)=\mathcal F^{-1}[R(A)e^{i\phi}].
\]

This is technically implementable and can suppress unstable low-amplitude phase coefficients. However:

1. Observed amplitude is not clean signal power or SNR. Strong backscatter can have high amplitude; weak useful texture can have low amplitude.
2. A learnable threshold is not a measured noise variance. Learned feature-space noise is neither necessarily additive nor circular Gaussian.
3. With GAP retained, RA skip attention still observes only the weighted DC coefficient. Its dependence on a spectrum-wide scale is dependence on magnitudes, not non-DC phase arrangement.
4. Full-spectrum RMS can be dominated by DC. The implementation offers an AC-only RMS ablation; neither choice is assumed superior.
5. L2-normalized Q/K make uniform per-channel attenuation ineffective. RA must change the relative spectral profile, which it does, to change the attention directions.
6. Replacing `angle/exp` with a stable complex formula is engineering, not a new research contribution. At exactly zero coefficients, RA returns zero whereas upstream PEM synthesizes a unit phasor. The tau-to-zero limit agrees only at nonzero coefficients.

The correct novelty claim is a particular placement and parameterization of shrinkage within the released Phaseformer, subject to experiments. “First reliability-aware Fourier attention” is unsupported.

### Selected architecture: NPS-Phaseformer

The proposed change is confined to the three active skip gates. Let p=P(X), z=p−GAP(p), and s=sqrt(GAP(z²)+epsilon²). Define

\[
d_0=\frac{\operatorname{GAP}(|z|)}{s},\qquad
 d_1=\frac{\operatorname{GAP}(|B_3*z|)}{s},
\]

where B3 is a 3×3 circular box average. Form D=[d0,d1] in B×2×C and compute

\[
\ell=\omega_{old}(\operatorname{GAP}(p))+\omega_{new}(D),\qquad Z=X\odot\sigma(\ell).
\]

`omega_new` is a bias-free Conv1D(2,1,k=3), initialized to zero. The old logit and its weights remain present. There are six additional parameters per active skip, or **18 total**. The nonlinear absolute value is applied before global averaging. The second descriptor distinguishes phase concentration after local aggregation; both are bounded summaries, not complete reconstructions of structure.

Only NPS is the primary recommendation. `ra_nps` applies RA to Q/K and NPS to the skips; it does **not** also change the skip extractor to RA. This choice separates the two diagnosed mechanisms cleanly. `ra_skip` and `ra_both` reproduce the handoff's skip placement as controls.

| Candidate | Mathematical case | Code / cost | Risk / expected outcome |
|---|---|---|---|
| RA Q/K | Low-SNR phase perturbation can be reduced | 48 learned scalars; elementwise spectral work; same FFT count | Plausible noise robustness; possible detail loss; accuracy unknown |
| RA skip only | Weighted DC statistic | 3 scalars | Does not fix phase pooling; weak primary contribution |
| NPS skip | Restores dependence on non-DC phase through nonlinear pooling | 18 weights; one local average + reductions at each active skip; no new FFT | Stronger diagnosis and easiest controlled test; potential benefit, not guaranteed |
| RA + NPS | Combines two independent hypotheses | 66 added weights | Test interaction; do not assume additive gains |
| FUSION-style second branch | Learns complementary magnitude restoration | More feature paths and tuning; must measure added cost | Larger overlap with prior work and less clean attribution |
| Physics-guided transmission/depth branch | Could model spatially varying degradation | Additional estimators/constraints, possible supervision | Potentially useful but misspecified physics can hurt; outside minimal feasible first study |

## 3. Theoretical / Mathematical Justification

### A. Exact DC collapse of the released skip descriptor

Use PyTorch's default forward DFT and 1/N inverse DFT. For one channel,

\[
p[n]=\frac1N\sum_k U[k]e^{i2\pi k\cdot n},\quad U[k]=e^{i\phi[k]}.
\]

Summing over spatial locations cancels every nonzero frequency:

\[
\operatorname{GAP}(p)=\frac{U[0]}{N}.
\]

For real input and nonzero DC, U[0]=sign(sum_n X[n]). Thus the descriptor is ±1/N, constant within any region of feature space where the DC sign does not change. At exactly zero DC, the implementation's `angle(0)=0` convention gives +1/N. The branch's descriptor gradient with respect to non-DC phase is zero in exact arithmetic. The full skip output still has its direct multiplicative gradient to X; this diagnosis does not imply the whole model is untrainable.

At large N and moderate convolution weights the sigmoid tends toward 0.5. This is not an absolute bound on learned gates: weights could grow very large. What is exact is the loss of non-DC phase information.

For RA, GAP(P_R)=R[0]U[0]/N. Although R[0] can depend on RMS of all amplitudes, the descriptor still cannot distinguish two spectra with equal magnitudes/DC and different non-DC phases.

### B. Why naive phase energy is also insufficient

Parseval gives, for a nonzero full unit-magnitude spectrum,

\[
\sum_n |p[n]|^2=\frac1N\sum_k |U[k]|^2=1.
\]

After DC subtraction the corresponding energy is 1−1/N, independent of non-DC phases. Therefore replacing GAP(p) with RMS(p) alone does not repair structural sensitivity. RMS is used only to normalize NPS; its numerators use nonlinear absolute statistics.

### C. NPS sensitivity and boundedness

Cauchy–Schwarz gives E|z|≤sqrt(Ez²), hence 0≤d0≤1. A normalized circular box filter is non-expansive in L2, so E|B3*z|≤sqrt(Ez²), giving 0≤d1≤1 as well (epsilon only reduces the ratios). Both descriptors are invariant to cyclic spatial shifts and, apart from epsilon, global positive rescaling of p. This makes their numerical scale less dependent on resolution than ±1/N.

A coherent unit spectrum produces a spatial impulse. Its centred L1/RMS ratio is

\[
d_0=\frac{2\sqrt{N-1}}{N},
\]

which approaches zero as N grows. A phase-scrambled, approximately Gaussian field has a ratio near sqrt(2/pi)≈0.798. Both can have identical Fourier magnitudes and identical energy. This supplies a constructive counterexample demonstrating that NPS can detect information unavailable to the old descriptor. It does not prove that an impulse is a desirable underwater feature, or that every useful structure is distinguishable using two numbers.

Because the correction is initialized to zero, the entire extended network initially equals the baseline, including loaded pretrained weights. Nevertheless, at initialization,

\[
\frac{\partial Z}{\partial\omega_{new}}=X\,\sigma'(\ell_{old})\,D
\]

in convolutional form, which is generally nonzero. Training can therefore learn the new route immediately. The parameterized function class includes the baseline. That guarantees representational inclusion, not improved optimization or generalization.

### D. RA noise argument and its limits

For S=A exp(i phi) and small circular complex noise N with E|N|²=sigma_n²,

\[
\Delta\phi\approx\frac{\operatorname{Im}(Ne^{-i\phi})}{A},\qquad
\operatorname{Var}(\Delta\phi)\approx\frac{\sigma_n^2}{2A^2}.
\]

Weak coefficients can have more uncertain phase. The approximation fails near A=0 and under strong/non-additive degradation. For scalar noisy signal estimation, minimizing (r−1)² sigma_s²+r² sigma_n² yields the familiar Wiener ratio r*=sigma_s²/(sigma_s²+sigma_n²). The proposed R resembles this ratio but substitutes observed amplitude for an unknown clean statistic; it is not the exact MMSE phase estimator.

Numerically the implementation computes

\[
U_R=\frac{F|F|}{|F|^2+\operatorname{softplus}(\theta)^2\,m_2+\epsilon^2},
\]

with m2 the full or AC mean spectral power. It avoids differentiating an explicit angle, handles zero input, uses full FFT symmetry, and keeps FFT work in FP32 under mixed precision. The scale statistic remains differentiable; it is not detached. Epsilon sets an intentional absolute floor and breaks exact scale invariance near zero.

### E. Attention stability is conditional

For normalized queries and keys, the first-order logit perturbation satisfies

\[
\|\Delta L\|\leq |T|\big(\|\Delta\hat Q\|\,\|\hat K\|+
\|\hat Q\|\,\|\Delta\hat K\|+\|\Delta\hat Q\|\,\|\Delta\hat K\|\big).
\]

The normalization derivative is approximately (I−qq^T/||q||²)/||q||, so suppressing almost all spectral energy can amplify directional instability after normalization. A bound on unnormalized Q error alone is insufficient. Measure normalized Q/K errors in corruption probes before claiming improved attention robustness.

### F. Complexity

Baseline FFT extraction costs O(CN log N). RA adds O(CN) elementwise work and reductions. NPS adds O(CN) local averaging and reductions only at the skips and reuses their existing phase reconstruction. Neither alters the attention map dimensions or the value path. Asymptotic equality does not imply equal wall-clock time. In particular, RA's complex multiplication can have quite different implementation cost from upstream `angle/exp` even with the same FFT count. The `unit_qk` algebra-only control computes F/|F| (and a unit phasor at zero) without reliability weighting, isolating that implementation effect. A faster RA runtime alone is not evidence for the reliability mechanism.

## 4. Comparative Results and Empirical Evaluation

### What has actually been verified

- Both supplied attachments were read in full; Figure 2 was rendered to check operation order.
- Five relevant repositories were cloned and source-inspected; FcaNet's attention source was additionally inspected.
- The official UIEB checkpoint was converted using a restricted class allowlist and checked against the released constructor's state-dict topology.
- 23 final tests passed: DC identity, phase sensitivity, cyclic shift invariance, near-zero/constant/random numerical cases, RA limit and symmetry, every variant's full backward pass, baseline-equivalent initialization, BF16 extractor execution and known metric cases.
- The synthetic DC identity error was 5.42e-19 in float64. NPS and the loaded baseline had maximum output difference 0.0 at zero initialization; all three new correction layers had nonzero gradients.
- The official checkpoint also ran on one real underwater sample; this is an unpaired execution demonstration only, stored in `results/pretrained_demo/`.
- The complete paired-data → train → save → reload → predict → score → compare pipeline ran on tiny synthetic data. Its scores are **software diagnostics only**, retained under `results/smoke/`.
- CPU parameters, partial operation counts and latency are measured in `results/benchmark_cpu.json`. A compact table is generated in `results/MEASURED_RESULTS.md`.

### Measured CPU comparison

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



### What has not been established

There is no retrained UIEB/UFO-120 baseline, no trained proposed underwater checkpoint, no new test PSNR/SSIM/UIQM improvement, no GPU latency measurement, and no reproduction of the paper's 13.0 GFLOPs. The four-loss `paper_fixed` path is implemented but requires VGG weights and has not undergone a full dataset training run. Full-model CUDA/FP16 behavior is not certified by the CPU BF16 extractor test.

| Evidence class | Baseline | NPS / RA variants |
|---|---|---|
| Published UIEB PSNR / SSIM / UIQM | 25.98 / .928 / 4.451 | Not applicable |
| New controlled UIEB test metrics | Pending dataset training | Pending dataset training |
| Official checkpoint available | Yes, converted UIEB weights included | Only zero-initialized compatible extension; not trained weights |
| Parameter count / CPU latency | Measured | Measured |
| Full benchmark FLOPs | Paper-reported only | Not measured; partial counts + FFT estimate supplied |

### Reproducible protocol

**Primary clean study:** freeze a new UIEB manifest with 90 test images and reserve 80 of the other 800 for validation; train on 720. This is deliberately labeled a new 720/80/90 protocol, not the paper's unidentified 800/90 split. Hash all files, reject missing pairs and cross-split exact duplicates, and never select checkpoints on test metrics. Near-duplicate scenes require an additional scene-level review; hashes alone cannot detect them.

Train all variants from scratch, seeds 123/456/789, identical initial shared weights, identical sample ordering and augmentation RNG, Adam 3e-4 with betas (.5,.999), cosine schedule for 200 epochs, 256px RGB inputs, batch size one and the same loss. These are an explicitly chosen study budget, not a recovered original recipe. After architecture/hyperparameter choice, an optional fixed-epoch 800/90 retraining can be a separate protocol, with the validation-selected epoch count frozen in advance.

The primary objective is .6 low-head + .4 high-head Charbonnier. This intentionally simple, shared objective isolates architecture. The secondary `paper_fixed` objective uses Charbonnier, Laplacian L1, 1−MS-SSIM and VGG16 feature L1 with the paper's reported final weights assigned to their named losses. It is a repaired **fixed-weight** protocol, not a reconstruction of the paper's dynamic weighting. Switching losses only for the proposed architecture would invalidate attribution.

Do not use the unknown-split pretrained checkpoint as initialization for the primary study: its training images may overlap the new test set. It is included for immediate inference and compatibility demonstrations. A separate fine-tuning study is defensible only on a verified disjoint target dataset and with equal baseline fine-tuning budgets.

### Controlled ablation matrix

| ID | CLI variant / setting | Question |
|---|---|---|
| B0 | baseline | Released architecture with shared repaired trainer |
| U0 | unit_qk | Does an algebra-only unit-phasor rewrite explain runtime savings? |
| R1 | ra_qk | Does the handoff reliability gate improve normalized channel attention? |
| R2 | ra_skip | Does weighted DC alone help? |
| R3 | ra_both | Original handoff combination |
| N1 | nps_l1 | Is one phase concentration statistic sufficient? |
| N2 | nps | Does local aggregation add benefit? Primary candidate |
| C1 | spatial_control | Matched 18-parameter correction using raw-feature statistics: is phase necessary? |
| C2 | ra_nps | Interaction of Q/K reliability and nonlinear skip pooling |
| R4 | ra_qk --fixed-tau | Is learning tau necessary? |
| R5 | ra_qk --scale ac | Does removing DC from threshold scaling matter? |

For the core 2×2 design B0/R1/N2/C2, calculate the interaction (C2−N2)−(R1−B0) for each metric rather than assuming additive contributions. Tune tau only on validation, for example {.01,.03,.1,.3}; report this tuning budget. Compare fixed and learned tau with equal seeds.

On held-out images evaluate clean inputs and fixed Gaussian noise levels 5/255, 10/255, 20/255. These are controlled perturbations, not faithful simulations of all underwater degradations. Add a separate blur test before claiming progress on the paper's deblurring failure cases. Evaluate edge error and crops to detect smoothing; inspect severe-colour-cast and low-contrast groups.

### Metrics and acceptance criteria

- PSNR: per-image RGB MSE, then average dB; no averaging MSE before the logarithm.
- SSIM: scikit-image, Gaussian sigma 1.5, population covariance, RGB channel average and range 255.
- Evaluate rounded/clipped lossless RGB PNG outputs; references are resized bicubically to the declared head resolution before scoring; no border crop. Validation checkpoint selection uses float RGB low-head PSNR, which is explicitly a different selection metric from quantized final reporting.
- UIQM: the bundled FUSION-compatible implementation, with missing SciPy import and flat-image divide-by-zero fixed. It retains historical trim indexing and blue UISM weight .144; report it as `uiqm_compat`, not as certified identical to the paper's unknown scorer. UIQM can reward excessive contrast or colour; inspect images alongside it.
- Parameter counts include all registered weights and a separate executed count. Original unused modules are retained identically.
- Timing: one image, FP32, both heads computed, inference mode, declared threads, warmup and repeated timings; synchronize CUDA before/after every timed call. File decoding and metric computation are excluded. Report device and CPU/GPU separately.
- Operation counts: conv + attention matmul use 2 FLOPs per MAC. FFT/IFFT are separately estimated at 10CN log2(N) real operations per pair. Pooling, normalization, nonlinearities and elementwise gates are excluded. Never label this sum as an exact full FLOP count or compare it directly with the paper's profiler result.

Accept NPS only if clean-test PSNR/SSIM improve consistently across the preregistered seeds without unacceptable visual/UIQM regressions and measured overhead is acceptable. Use paired image bootstrap intervals plus a separate mean/std across training seeds. Image bootstrap does not capture training uncertainty. Report negative results if C1 matches NPS, if RA merely oversmooths, or if NPS does not beat B0. No numerical performance gain is predicted in advance.

## References and source links

1. Khan et al., Phaseformer, WACV 2025: https://openaccess.thecvf.com/content/WACV2025/html/Khan_Phaseformer_Phase-Based_Attention_Mechanism_for_Underwater_Image_Restoration_and_Beyond_WACV_2025_paper.html
2. Official Phaseformer code: https://github.com/Mdraqibkhan/Phaseformer/tree/6e759dc782e514c985672c545205dea86a3d21b5
3. Walia et al., FUSION, CVPR Workshops 2025: https://arxiv.org/html/2504.01243v2 ; code https://github.com/shravan-18/FUSION/tree/ccfd2fd0eb7a7e7e77aa83b6bdbb9df8cf01f5e5
4. Spectroformer code: https://github.com/Mdraqibkhan/Spectroformer/tree/0a0c17b37e478bcc786cf923698f43c999587de9
5. UIEPTA code: https://github.com/Mdraqibkhan/UIEPTA/tree/a790eb8c15582720e48b77819c60386e4d5e973a
6. FGDPA code: https://github.com/LethyZhang/FGDPA/tree/530c692feb99ecf6205bb929118a9167e2809d4d
7. Qin et al., FcaNet, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.html ; code https://github.com/cfzd/FcaNet
8. Kovesi's phase congruency implementation/documentation: https://peterkovesi.github.io/ImagePhaseCongruency.jl/dev/
9. Upstream dynamic-weight issue: https://github.com/Mdraqibkhan/Phaseformer/issues/7
10. Upstream test-split issue: https://github.com/Mdraqibkhan/Phaseformer/issues/6
11. UIEB official data page: https://li-chongyi.github.io/proj_benchmark.html
12. UFO-120 official data page: https://irvlab.cs.umn.edu/resources/ufo-120-dataset

The supplied handoff was used as a proposal to test, not an authoritative source of verified results. The attached 12-page paper was the source for published scores. Source code observations are pinned to the checked commits; later upstream edits may change them.
