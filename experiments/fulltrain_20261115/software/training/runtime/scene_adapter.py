"""Portable inference for the fixed ten-epoch scene-only comparator.

Inputs are the same unit-normalized frozen image/text embeddings used by the
training study. This module does not load an image encoder or regional features.
"""
import torch
from torch.nn import functional as F


class SceneResidualAdapter(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dimension, 128), torch.nn.GELU(),
            torch.nn.Linear(128, dimension),
        )
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def encode(self, scene_embeddings):
        return F.normalize(scene_embeddings + self.scale * self.net(scene_embeddings), dim=-1)

    def forward(self, queries, scene_embeddings):
        return queries @ self.encode(scene_embeddings).T


def load_scene_adapter(path, device="cpu", expected_backbone=None):
    # The measured checkpoint metadata contains torch.__version__, whose type
    # is this inert string subclass. No unrestricted pickle loading is used.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["variant"] != "scene_adapter" or checkpoint["epoch"] != 10:
        raise ValueError("Expected the fixed ten-epoch scene-adapter checkpoint")
    if expected_backbone is not None and metadata["backbone"] != expected_backbone:
        raise ValueError("Backbone does not match checkpoint")
    state = checkpoint["state_dict"]
    dimension = state["net.0.weight"].shape[1]
    head = SceneResidualAdapter(dimension)
    head.load_state_dict(state, strict=True)
    if sum(p.numel() for p in head.parameters()) != metadata["parameters"]:
        raise ValueError("Parameter count does not match checkpoint")
    head.eval().to(device)
    head.requires_grad_(False)
    return head
