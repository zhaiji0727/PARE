"""Verbatim shared residual scene/region adapter class."""
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
