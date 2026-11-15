"""Compose structural ABC switches with existing matched information controls.

Software preparation only: no selected path, experiment protocol or run entry.
Inherited model/factory/wrapper files stay byte-identical to their references.
"""
from .paths import PathIsolatedJointPARE

PATHS = [f'{n:03b}' for n in range(8)]
CONTROLS = ['full', 'constant', 'no_token', 'uniform', 'stats_off']
CONTROL_VARIANT = {'full': 'joint_full', 'constant': 'joint_constant',
                   'no_token': 'joint_no_token', 'uniform': 'joint_uniform',
                   'stats_off': 'joint_constant'}
CONTROL_FLAGS = {'full': set(), 'constant': set(), 'no_token': {'ty'},
                 'uniform': {'sr'}, 'stats_off': {'sr'}}


def build_path_control(dimension, abc, control, coarse_mean, fine_mean):
    if abc not in PATHS:
        raise ValueError('Expected explicit three-bit ABC path')
    if control not in CONTROLS:
        raise ValueError('Unknown matched control: ' + str(control))
    model = PathIsolatedJointPARE(dimension, coarse_mean, fine_mean,
        fine_attention=abc[0] == '1', fine_gate_input=abc[1] == '1',
        coarse_cfar_feature=abc[2] == '1')
    # Every relevant JointPARE variant constructs the same parameters/buffers
    # in the same order. Its variant selects existing descriptor transforms.
    model.variant = CONTROL_VARIANT[control]
    # Never assign a replacement set: the wrapper's A=0 'cfar' flag must survive.
    model.physics.ablate.update(CONTROL_FLAGS[control])
    model.requested_abc = abc
    model.requested_control = control
    return model
