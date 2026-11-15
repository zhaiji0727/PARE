"""Unbound three-path control wrapper around the unchanged JointPARE.

No training runner or experiment protocol. All three flags mean path enabled.
A: fine attended vector; B: fine content/count in the gate aggregate;
C: fourth coarse descriptor coordinate (regional CFAR fraction).
The original two sigmoid gates and every trainable parameter are unchanged.
"""
from joint_model import JointPARE


class PathIsolatedJointPARE(JointPARE):
    def __init__(self, dimension, coarse_mean, fine_mean, *,
                 fine_attention=True, fine_gate_input=True, coarse_cfar_feature=True):
        switches = (fine_attention, fine_gate_input, coarse_cfar_feature)
        if any(type(value) is not bool for value in switches):
            raise TypeError("All three path switches must be explicit booleans")
        # Preserve original construction order, registration, initialization,
        # adapter, physics parameters, and full-model forward method.
        super().__init__(dimension, 'joint_full', coarse_mean, fine_mean)
        self.paths_enabled = dict(A=fine_attention, B=fine_gate_input, C=coarse_cfar_feature)
        if not fine_attention:
            # Existing SAQRE flag replaces f_att with scene. It deliberately
            # does not disable fine input to agg or the fourth coarse feature.
            self.physics.ablate.add('cfar')

    def adapt_docs(self, docs):
        adapted = super().adapt_docs(docs)
        if not self.paths_enabled['B']:
            # Exactly the expression used inside SAQREHead's cfar_strict path.
            # The numerator AND denominator exclude fine regions.
            coarse = adapted['c_emb'].masked_fill(~adapted['c_mask'].unsqueeze(-1), 0.0)
            adapted['agg'] = (adapted['g'] + coarse.sum(1)) / (
                1 + adapted['c_mask'].sum(1, keepdim=True))
        if not self.paths_enabled['C']:
            # Never mutate the caller's cache tensor; retain its other coords.
            adapted['c_tf'] = adapted['c_tf'].clone()
            adapted['c_tf'][..., 3] = 0.0
        return adapted
