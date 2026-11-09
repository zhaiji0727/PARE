"""One controlled intervention: remove regional variation only at h_psi input."""
import torch

from repair_head import RepairConfig, RepairHead


VARIANTS = ('conditional', 'constant_descriptor', 'token_off')


def training_descriptor_means(cache, train_ids):
    """CPU float64 arithmetic means over valid crops of actual training IDs.

    Cache order and within-image crop order are retained. Padding and every
    non-training image are excluded before values are inspected or accumulated.
    """
    train_ids = list(train_ids)
    if len(train_ids) != len(set(train_ids)) or not train_ids:
        raise ValueError('Training IDs must be nonempty and unique')
    ids = cache['ids']
    if len(ids) != len(set(ids)) or not set(train_ids) <= set(ids):
        raise ValueError('Missing or duplicate cache identities')
    wanted = set(train_ids)
    rows = torch.tensor([i for i, name in enumerate(ids) if name in wanted], dtype=torch.long)
    result = {'n_training_images': len(rows), 'arithmetic': 'CPU float64 sum/count; cast once to float32',
              'crop_weighting': 'one equal weight per valid crop', 'branches': {}}
    for label, prefix in [('coarse', 'c'), ('fine', 'f')]:
        features = cache[prefix + '_tf'][rows].detach().cpu()
        counts = cache[prefix + '_n'][rows].detach().cpu()
        if features.ndim != 3 or features.shape[-1] != 4 or not torch.is_floating_point(features):
            raise ValueError('Expected floating point N x M x 4 descriptors')
        if counts.shape != (len(rows),) or bool(((counts < 0) | (counts > features.shape[1])).any()):
            raise ValueError('Invalid crop counts')
        if not torch.equal(counts, counts.long()):
            raise ValueError('Crop counts must be integers')
        mask = torch.arange(features.shape[1])[None, :] < counts[:, None]
        valid = features[mask].to(torch.float64)
        if not len(valid) or not bool(torch.isfinite(valid).all()):
            raise ValueError('A branch must contain finite valid training descriptors')
        mean = valid.sum(dim=0) / len(valid)
        result['branches'][label] = {'n_valid_crops': len(valid), 'mean_float64': mean.tolist(),
                                     'mean_float32': mean.float().tolist()}
    return result


class TokenMechanismHead(RepairHead):
    """Preserve the old B0 head bytewise; only the token input may differ.

    Every variant retains all original parameter tensors. The two mean buffers
    are identical across variants, non-trainable, and consume no random draws.
    Conditional and token_off paths do not use these buffers in computation.
    """
    MEAN_BUFFERS = ('constant_coarse_descriptor', 'constant_fine_descriptor')

    def __init__(self, dim, variant, means, hidden=128):
        if variant not in VARIANTS:
            raise ValueError('Unknown token intervention')
        config = RepairConfig(A=1, B=0, C=1, T=int(variant != 'token_off'),
                              fusion='original', reliability='original', token_mode='original')
        super().__init__(dim, config, hidden)
        self.variant = variant
        for branch, name in zip(('coarse', 'fine'), self.MEAN_BUFFERS):
            mean = torch.tensor(means['branches'][branch]['mean_float32'], dtype=torch.float32)
            if mean.shape != (4,) or not bool(torch.isfinite(mean).all()):
                raise ValueError('Expected finite four-dimensional means')
            self.register_buffer(name, mean)

    def load_base_state_dict(self, state):
        missing = set(self.NEW_PARAMETER_NAMES) | set(self.MEAN_BUFFERS)
        if set(state) != set(self.state_dict()) - missing:
            raise ValueError('Original base parameter schema differs')
        loaded = self.load_state_dict(state, strict=False)
        if set(loaded.missing_keys) != missing or loaded.unexpected_keys:
            raise ValueError('Unexpected base state mismatch')
        return loaded

    def legacy_parameter_state(self):
        return {key: value for key, value in self.state_dict().items() if key not in self.MEAN_BUFFERS}

    def fuse_batch(self, q, docs):
        if self.variant == 'constant_descriptor':
            # c_tf/f_tf feed only token augmentation in the fixed A1/B0/C1 head.
            # All mu/t/nu values, embeddings, crop masks and gate inputs survive.
            docs = dict(docs)
            for prefix, name in zip(('c', 'f'), self.MEAN_BUFFERS):
                mean = getattr(self, name)
                original = docs[prefix + '_tf']
                if mean.dtype != original.dtype or mean.device != original.device:
                    raise ValueError('Descriptor means and cached inputs must share float32 device')
                docs[prefix + '_tf'] = mean.expand_as(original)
        return super().fuse_batch(q, docs)
