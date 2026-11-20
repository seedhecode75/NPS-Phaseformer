# Code-level change map

| Original site | New site | Exact operation |
|---|---|---|
| `model_with_eca.py::inv_mag` | `phaseformer_ext/modules.py::ReliabilityPhase` | Optional Q/K or skip extractor; complex `F*abs(F)/(power+tau²)` with numerical floor |
| `model_with_eca.py::MDTA.forward` Q/K at lines 76–77 | `ReliableMDTA.forward` | Replace only the Q/K phase extractor; retain qkv projections, reshapes, L2 normalization, temperature and V path |
| `model_with_eca.py::ECA.forward` | `PhaseStatisticSkip.forward` / `phase_descriptors` | Preserve old logit, add zero-initialized nonlinear phase descriptor correction |
| `model_with_eca.py::Restormer.attention[0:3]` | `phaseformer_ext/model.py::build_model` | Replace exactly three active skips; retain unused fourth skip and `ups2` for matching parameter accounting |
| `model_with_eca.py::Restormer.encoders/decoders/refinement[*].attn` | `build_model` | Recursively replace all 24 MDTA instances only for RA-Q/K variants |
| Released `train.py` | root `train.py`, `phaseformer_ext/losses.py` | New shared trainer, correct losses/ranges, explicit configuration, reproducible manifests, safe state checkpoints |
| Released `data.py`/`dataset.py` | `phaseformer_ext/data.py`, `scripts/prepare_data.py` | Verified pairing, split freezing, coherent image transforms |
| Released `test.py`/`imqual_utils.py` | `evaluate.py`, `phaseformer_ext/metrics.py` | Declare output head/resolution and metric definitions; write per-image CSVs |

`legacy/` preserves the original Python files byte-for-byte. Root `model_with_eca.py` removes only unused `typing_extensions.Self` and `cv2` imports; no forward mathematics is edited. Extensions reuse the original trainable submodules so baseline checkpoint names remain stable. `load_initial_weights` rejects all unexpected keys and missing keys other than deliberately added parameters.

The original default ECA declarations use `ch=[16,16,32,64]`, while active features arrive with channels 64,32,16. For these values all constructed kernels happen to be size 3, so this mismatch is not repaired in only one model. The fourth ECA and `ups2` are never called by the released forward function.

The four-assignment-section report explains the distinction between paper-faithful concepts, released-code fidelity and the newly controlled training protocol.
