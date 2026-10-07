"""Exercise the evaluation/report flow with synthetic embeddings, not model metrics."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


class FakeModel(torch.nn.Module):
    def encode_text(self, tokens):
        return tokens


class FakeHead(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()

    def fuse_batch(self, query, docs):
        return docs["g"]


@pytest.mark.parametrize("script", ["eval_retrieval_phys", "eval_retrieval_baselines"])
@pytest.mark.parametrize("flags,expected", [([], 3), (["--max_queries", "2"], 2),
                                            (["--query_mode", "all-captions"], 4)])
def test_evaluators_record_query_ranks_and_coverage(tmp_path, monkeypatch, script, flags, expected):
    spec = importlib.util.spec_from_file_location(script, ROOT / "code" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    cache = {"ids": ["A_1", "B_2", "C_3"], "D": 2, "Mc": 1, "Mf": 1,
             "scene": scene, "agg": scene.clone(), "nu": torch.zeros(3)}
    for prefix in ("c", "f"):
        cache.update({f"{prefix}_emb": scene[:, None, :].clone(),
                      f"{prefix}_n": torch.ones(3, dtype=torch.long),
                      f"{prefix}_mu": torch.zeros(3, 1),
                      f"{prefix}_t": torch.zeros(3, 1),
                      f"{prefix}_tf": torch.zeros(3, 1, 4)})
    cache_path, head_path, csv_path, out = [tmp_path / name for name in ("cache.pt", "head.pt", "captions.csv", "result.json")]
    torch.save(cache, cache_path)
    torch.save({"state_dict": {}, "D": 2, "ablate": []}, head_path)
    csv_path.write_text("Image Name, Caption\nA_1,east\nA_1,east\nB_2,north\nC_3,east\n")
    monkeypatch.setattr(module, "create_backbone", lambda *args: (FakeModel(), None, None))
    monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module.open_clip, "get_tokenizer", lambda name: lambda texts: torch.tensor(
        [[0., 1.] if text == "north" else [1., 0.] for text in texts]))
    verified = []

    def fake_validate(cache, *args, **kwargs):
        verified.append("cache")
        assert not kwargs["allow_legacy"]
        return {"synthetic_test_signature": True}

    monkeypatch.setattr(module, "validate_cache", fake_validate)
    argv = [script, "--cache", str(cache_path), "--captions", str(csv_path),
            "--candidate_K", "2", "--hard_M", "2", "--pools", "2", "--out", str(out), *flags]
    if script == "eval_retrieval_phys":
        monkeypatch.setattr(module, "SAQREHead", FakeHead)
        monkeypatch.setattr(module, "validate_head", lambda *args, **kwargs: verified.append("head"))
        argv += ["--head", str(head_path)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    result = json.loads(out.read_text())
    assert result["n_queries"] == expected
    assert len(result["queries"]) == expected
    assert result["query_protocol"]["evaluated_queries"] == expected
    assert result["gallery_ids"] == cache["ids"]
    assert result["provenance"]["cache_signature"] == {"synthetic_test_signature": True}
    assert "cache" in verified
    for record in result["queries"]:
        assert len(record["controlled_random_pool_ranks"]["scene"]) == 2
        assert record["image_id"] == cache["ids"][record["gold_gallery_row"]]
    if script == "eval_retrieval_phys":
        assert "head" in verified
        if expected > 2:
            missing = next(record for record in result["queries"] if record["image_id"] == "C_3")
            assert not missing["natural_candidate_included"]
            assert missing["natural_PARE_rank"] is None
