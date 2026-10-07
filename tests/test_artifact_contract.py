import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import artifacts
from artifacts import (CODE, extraction_signature, file_sha256, object_sha256,
                       parse_ablation_flags, validate_cache, validate_head)


@pytest.fixture(autouse=True)
def fake_pretrained_weights(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test-backbone-weights")
    original = artifacts.resolve_pretrained_path
    monkeypatch.setattr(artifacts, "resolve_pretrained_path", lambda model, name:
                        weights if name == "test-weights" else original(model, name))


def cache(mode="physics"):
    signature = extraction_signature("test-model", "test-weights", primitive_mode=mode)
    ids = ["one.png", "two.png"]
    return {"ids": ids, "D": 16, "Mc": 6, "Mf": 8,
            "provenance": {"schema_version": 2, "signature": signature,
                           "signature_sha256": object_sha256(signature),
                           "ids_sha256": object_sha256(ids)}}


def test_current_cache_contract_and_head_match():
    value = cache()
    signature = validate_cache(value, "test-model", "test-weights")
    head = {"D": 16, "provenance": {"cache_signature": signature,
            "head_source_sha256": file_sha256(CODE / "saqre_phys.py")}}
    validate_head(head, value)
    with pytest.raises(ValueError, match="different encoder"):
        validate_head(head, cache("uniform_grid"))


def test_legacy_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="Legacy"):
        validate_cache({}, "test-model", "test-weights")
    with pytest.warns(UserWarning):
        assert validate_cache({}, "test-model", "test-weights", allow_legacy=True) is None
    with pytest.raises(ValueError, match="Unversioned"):
        validate_head({}, {})


def test_changed_ids_encoder_code_or_signature_are_rejected():
    value = cache()
    with pytest.raises(ValueError, match="Stale"):
        validate_cache(value, "other-model", "test-weights")
    changed = copy.deepcopy(value)
    changed["ids"].reverse()
    with pytest.raises(ValueError, match="image IDs"):
        validate_cache(changed, "test-model", "test-weights")
    changed = copy.deepcopy(value)
    changed["provenance"]["signature"]["source_sha256"]["saqre_phys.py"] = "old"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_cache(changed, "test-model", "test-weights")
    changed["provenance"]["signature_sha256"] = object_sha256(changed["provenance"]["signature"])
    with pytest.raises(ValueError, match="Stale"):
        validate_cache(changed, "test-model", "test-weights")


def test_ablation_names_and_physics_free_contract():
    assert parse_ablation_flags("sr,ty,cfar_strict", cache()) == {"sr", "ty", "cfar_strict"}
    with pytest.raises(ValueError, match="Unknown"):
        parse_ablation_flags("crar")
    with pytest.raises(ValueError, match="not both"):
        parse_ablation_flags("cfar,cfar_strict")
    with pytest.raises(ValueError, match="requires both"):
        parse_ablation_flags("", cache("uniform_grid"))
    assert parse_ablation_flags("sr,ty", cache("uniform_grid")) == {"sr", "ty"}


def test_replacing_weights_at_same_path_changes_signature(tmp_path):
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"first-version")
    before = extraction_signature("local-model", str(weights))
    weights.write_bytes(b"second-version")
    after = extraction_signature("local-model", str(weights))
    assert before["pretrained_weight_sha256"] != after["pretrained_weight_sha256"]
    assert before != after
