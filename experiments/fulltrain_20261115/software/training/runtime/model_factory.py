"""One existing-control composition; the upstream JointPARE stays unchanged."""
from joint_model import JointPARE


def build_joint_model(dimension, variant, coarse_mean, fine_mean):
    if variant == 'joint_stats_off':
        # Existing constant-descriptor transform retains scene/coarse/fine paths,
        # region masks/content, trainable token capacity and parameter order.
        model = JointPARE(dimension, 'joint_constant', coarse_mean, fine_mean)
        # Existing uniform-reliability flag sets both branches' valid rho to1.
        model.physics.ablate.add('sr')
        return model
    # All original variants take precisely the original constructor/code path.
    return JointPARE(dimension, variant, coarse_mean, fine_mean)
