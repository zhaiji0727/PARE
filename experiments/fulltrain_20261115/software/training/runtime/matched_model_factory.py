"""Dispatch explicit model specs; existing variant definitions are unchanged."""
from model_spec import parse_model_id
from model_factory import build_joint_model
from path_control_factory import build_path_control


def build_matched_model(dimension, model_id, coarse_mean, fine_mean):
    spec = parse_model_id(model_id)
    if spec['family'] == 'original_variant':
        return build_joint_model(dimension, spec['original_variant'], coarse_mean, fine_mean)
    return build_path_control(dimension, spec['abc'], spec['control'], coarse_mean, fine_mean)
