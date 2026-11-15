"""Exact state restoration helpers. No data loading or experiment launch."""
from contextlib import contextmanager
import copy
import hashlib
import json
import random

import torch


def order_sha(order):
    return hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()


def assert_same(a, b, label="state"):
    """Tensor equality includes shape/dtype/bytes, but not storage location."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise ValueError(f"{label}: tensor type differs")
        aa, bb = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        if aa.dtype != bb.dtype or aa.shape != bb.shape:
            raise ValueError(f"{label}: tensor schema differs")
        if not torch.equal(aa.reshape(-1).view(torch.uint8), bb.reshape(-1).view(torch.uint8)):
            raise ValueError(f"{label}: tensor bytes differ")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError(f"{label}: mapping keys differ")
        for key in a:
            assert_same(a[key], b[key], f"{label}.{key}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError(f"{label}: sequence schema differs")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{label}[{i}]")
    elif a != b:
        raise ValueError(f"{label}: values differ")


def validate_epoch_state(checkpoint, seed, n_pairs, epoch):
    required = {"state_dict", "optimizer", "metadata", "history", "order",
                "python_rng", "torch_rng", "cuda_rng"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Missing checkpoint states: {required - set(checkpoint)}")
    if checkpoint["metadata"]["seed"] != seed:
        raise ValueError("Checkpoint seed differs")
    history = checkpoint["history"]
    if len(history) != epoch or [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint history is not the exact required epoch prefix")
    expected_order = list(range(n_pairs))
    expected_rng = random.Random(seed)
    for row in history:
        expected_rng.shuffle(expected_order)
        if row["pair_order_sha256"] != order_sha(expected_order):
            raise ValueError("Checkpoint has unexpected historical pair order")
    if checkpoint["order"] != expected_order:
        raise ValueError("Saved next-epoch starting order differs")
    if checkpoint["python_rng"] != expected_rng.getstate():
        raise ValueError("Saved Python shuffle RNG differs")
    state = checkpoint["torch_rng"]
    if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("Invalid CPU RNG state")


def restore(checkpoint, model, optimizer, *, seed, n_pairs, epoch, device):
    """Call after all construction/loading; restore RNG last, including shuffle."""
    validate_epoch_state(checkpoint, seed, n_pairs, epoch)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    assert_same(checkpoint["state_dict"], model.state_dict(), "restored_model")
    assert_same(checkpoint["optimizer"], optimizer.state_dict(), "restored_optimizer")
    rng = random.Random()
    rng.setstate(checkpoint["python_rng"])
    order = list(checkpoint["order"])
    history = copy.deepcopy(checkpoint["history"])
    model.train()
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if torch.device(device).type == "cuda":
        if not torch.is_tensor(checkpoint["cuda_rng"]):
            raise ValueError("CUDA continuation requires saved CUDA RNG state")
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device=device)
    elif checkpoint["cuda_rng"] is not None:
        raise ValueError("Refusing to move a CUDA trajectory to CPU")
    assert_same(torch.get_rng_state(), checkpoint["torch_rng"], "restored_cpu_rng")
    if torch.device(device).type == "cuda":
        assert_same(torch.cuda.get_rng_state(device), checkpoint["cuda_rng"], "restored_cuda_rng")
    return order, rng, history


def capture(model, optimizer, metadata, history, order, rng, device):
    """Snapshot legacy-compatible state, cloning to avoid later mutation."""
    return {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "metadata": copy.deepcopy(metadata),
        "history": copy.deepcopy(history),
        "order": list(order),
        "python_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(device).clone() if torch.device(device).type == "cuda" else None,
    }


@contextmanager
def preserve_rng_for_evaluation(rng, device):
    """Intermediate scoring must not affect the remaining training trajectory."""
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if torch.device(device).type == "cuda" else None
    local = rng.getstate()
    global_python = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device=device)
        rng.setstate(local)
        random.setstate(global_python)
