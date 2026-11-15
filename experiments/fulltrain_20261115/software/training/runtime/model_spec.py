"""Standard-library-only explicit names for original models and matched controls."""
ORIGINAL_VARIANTS = ['adapter', 'original_full', 'joint_full', 'joint_constant', 'joint_no_token',
                     'joint_uniform', 'joint_paths_off', 'joint_scene_control', 'joint_stats_off']
PATHS = [f'{n:03b}' for n in range(8)]
CONTROLS = ['full', 'constant', 'no_token', 'uniform', 'stats_off']
VARIANTS = ORIGINAL_VARIANTS + ['path_'+abc for abc in PATHS] + [
    'path_'+abc+'__'+control for abc in PATHS for control in CONTROLS]


def parse_model_id(value):
    if value in ORIGINAL_VARIANTS:
        return {'family': 'original_variant', 'abc': None, 'control': None, 'original_variant': value}
    if value in ['path_'+abc for abc in PATHS]:
        return {'family': 'matched_path', 'abc': value[5:], 'control': 'full', 'original_variant': None}
    if value in VARIANTS:
        abc, control = value[5:].split('__')
        return {'family': 'matched_path', 'abc': abc, 'control': control, 'original_variant': None}
    raise ValueError('Unknown model identifier: '+str(value))


def canonical_model_key(value):
    spec = parse_model_id(value)
    return spec['family'], spec['original_variant'], spec['abc'], spec['control']
