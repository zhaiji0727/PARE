import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repaired_runner", ROOT / "scripts/run_repaired_experiment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_smoke_plan_is_separate_fit_subset_and_four_variants(tmp_path):
    plan = module.build_plan(ROOT, tmp_path / "fresh", smoke=True)
    assert len(plan) == 17
    assert not any(step["stage"] == "sweep" for step in plan)
    train = [step for step in plan if step["stage"] == "train"]
    assert {step["variant"] for step in train} == set(module.VARIANTS)
    for step in plan:
        assert "--allow_legacy_artifacts" not in step["command"]
        if step["stage"] == "cache" and step["side"] == "train":
            assert step["ids_path"].endswith("ret_clean_20261007/fit_ids.json")
        if step["stage"] == "train":
            assert step["command"][step["command"].index("--max_pairs") + 1] == "32"
    evaluations = [step for step in plan if step["stage"] == "eval" and "k" in step]
    assert {step["k"] for step in evaluations} == {8, 16}


def test_formal_plan_requires_selection_and_all_queries(tmp_path):
    plan = module.build_plan(ROOT, tmp_path / "fresh")
    assert len(plan) == 34
    sweep = next(step for step in plan if step["name"] == "sweep")
    assert "--fit_ids" in sweep["command"] and "--val_ids" in sweep["command"]
    assert sweep["command"][sweep["command"].index("--epochs") + 1] == "10"
    assert sweep["command"][sweep["command"].index("--val_q") + 1] == "0"
    for step in plan:
        if step["stage"] == "train":
            assert "sweep" in step["requires"]
            assert "<selected_lr>" in step["command"]
        if step["stage"] == "eval":
            assert step["command"][step["command"].index("--max_queries") + 1] == "0"
    assert {step["k"] for step in plan if "k" in step} == {100, 200, 500, 1000, 2000, 3688}


def test_completed_steps_verify_hash_and_refuse_mutation(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("{}")
    step = {"name": "check", "outputs": [str(path)], "status": "complete",
            "artifact_sha256": {str(path): module.sha256(path)}}
    module.verify_completed(step)
    path.write_text("{\"changed\": true}")
    with pytest.raises(ValueError, match="changed"):
        module.verify_completed(step)


def test_formal_selection_cannot_skip_sweep(tmp_path):
    with pytest.raises(ValueError, match="not complete"):
        module.selected_parameters(tmp_path, {"sweep": {"name": "sweep", "status": "pending"}})


def test_smoke_environment_is_cpu_offline_and_bounded(tmp_path):
    env = module.environment(tmp_path, tmp_path / "out", "cpu")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"


def test_nonfinite_metrics_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        module.finite_json({"ranks": [1, float("nan")]})


def fake_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    split = workspace / "data/ret_clean_20261007"
    split.mkdir(parents=True)
    (split / "manifest.json").write_text(json.dumps({
        "content_disjoint_train_test": True, "content_disjoint_fit_val": True,
        "official_test_duplicate_content_group_count": 0, "output_sha256": {},
        "split_counts": {"test": 3688}}))
    code = workspace / "code"
    code.mkdir()
    for name in module.SOURCE_FILES:
        (code / name).write_text("# Synthetic dry-run fixture\n")
    return workspace


def test_dry_run_creates_no_outputs(tmp_path, monkeypatch, capsys):
    workspace = fake_workspace(tmp_path)
    output = tmp_path / "new_output"
    monkeypatch.setattr(sys, "argv", ["runner", "--workspace", str(workspace),
                                      "--out", str(output), "--dry-run"])
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] and not result["paper_results"]
    assert len(result["steps"]) == 34
    assert not output.exists()


def test_existing_unowned_directory_never_overwritten(tmp_path, monkeypatch):
    workspace = fake_workspace(tmp_path)
    output = tmp_path / "already_exists"
    output.mkdir()
    sentinel = output / "result.pt"
    sentinel.write_bytes(b"existing user result")
    monkeypatch.setattr(sys, "argv", ["runner", "--workspace", str(workspace), "--out", str(output)])
    with pytest.raises(ValueError, match="Refusing to reuse"):
        module.main()
    assert sentinel.read_bytes() == b"existing user result"
    assert list(output.iterdir()) == [sentinel]
