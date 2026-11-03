"""Portable paths and checked model loading for the recovered experiment code."""
import os
from pathlib import Path


ROOT = Path(os.environ.get("RS_RAG_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get("RS_RAG_DATA_ROOT", os.environ.get("DATA_ROOT", ROOT / "data")))
MODEL_ROOT = Path(os.environ.get("RS_RAG_MODEL_ROOT", os.environ.get("MODEL_ROOT", ROOT / "models")))
RESULTS_ROOT = Path(os.environ.get("RS_RAG_RESULTS_ROOT", ROOT / "results"))


def get_device():
    import torch

    requested = os.environ.get("PARE_DEVICE", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PARE_DEVICE requests CUDA, but CUDA is unavailable")
    return requested


def load_checkpoint_strict(model, path):
    """Reject incomplete or unrelated external CLIP checkpoints."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dictionary in {path}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def create_backbone(model_name, pretrained, checkpoint=None):
    import open_clip

    model, train_transform, eval_transform = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=str(MODEL_ROOT),
    )
    if checkpoint:
        load_checkpoint_strict(model, checkpoint)
    return model, train_transform, eval_transform
