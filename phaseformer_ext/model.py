import torch
from model_with_eca import Restormer, MDTA
from .modules import ReliableMDTA, ReliabilityPhase, UnitPhase, ReliabilitySkip, PhaseStatisticSkip

VARIANTS = ('baseline', 'unit_qk', 'ra_qk', 'ra_skip', 'ra_both', 'nps', 'nps_l1', 'spatial_control', 'ra_nps')

def build_model(variant='baseline', tau=0.1, learnable=True, scale='full'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant}')
    model = Restormer()
    opts = dict(tau=tau, learnable=learnable, scale=scale)
    if variant in ('unit_qk', 'ra_qk', 'ra_both', 'ra_nps'):
        for parent in list(model.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, MDTA):
                    replacement = ReliableMDTA(child, **opts)
                    if variant == 'unit_qk':
                        replacement.q_phase, replacement.k_phase = UnitPhase(), UnitPhase()
                    setattr(parent, name, replacement)
    for i in range(3):  # attention[3] is unused in the released forward pass.
        original = model.attention[i]
        if variant in ('nps', 'nps_l1', 'spatial_control', 'ra_nps'):
            model.attention[i] = PhaseStatisticSkip(original,
                descriptor='spatial' if variant == 'spatial_control' else 'phase',
                statistics='l1' if variant == 'nps_l1' else 'both')
        elif variant in ('ra_skip', 'ra_both'):
            model.attention[i] = ReliabilitySkip(original, ReliabilityPhase(**opts))
    return model


def load_initial_weights(model, checkpoint):
    """Allow ONLY deliberately added parameters to be absent from a baseline."""
    obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = obj.get('model', obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = ('.correction.weight', '.q_phase.raw_tau', '.k_phase.raw_tau', '.extractor.raw_tau')
    if unexpected or any(not k.endswith(allowed) for k in missing):
        raise ValueError(f'Checkpoint architecture mismatch: {missing=}, {unexpected=}')
    return {'missing_new_parameters': missing, 'unexpected': unexpected}
