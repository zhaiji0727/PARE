import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retrieval_queries", ROOT / "code/retrieval_queries.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_default_covers_all_and_legacy_limit_restores_order():
    captions = {f"SOURCE_{i}": [str(i), "other"] for i in range(2100)}
    records, protocol = module.select_queries(captions, list(captions))
    assert len(records) == 2100
    assert not protocol["truncated"]
    legacy, limited = module.select_queries(captions, list(captions), max_queries=2000)
    assert legacy == records[:2000]
    assert len(limited["gallery_images_without_evaluated_query"]) == 100


def test_all_captions_preserves_duplicate_rows_and_query_ids():
    records, protocol = module.select_queries({"A_1": ["same", "same"], "B_2": ["third"]},
                                             ["B_2", "A_1"], "all-captions")
    assert len(records) == 3
    assert len({record["query_id"] for record in records}) == 3
    assert [record["gold_gallery_row"] for record in records] == [1, 1, 0]
    assert protocol["query_prefix_counts"] == {"A": 2, "B": 1}
    assert protocol["query_image_prefix_counts"] == {"A": 1, "B": 1}


def test_query_cli_aliases_and_zero_default():
    parser = argparse.ArgumentParser()
    module.add_query_arguments(parser)
    assert parser.parse_args([]).max_q == 0
    assert parser.parse_args(["--max_q", "2000"]).max_q == 2000
    assert parser.parse_args(["--max_queries", "2000"]).max_q == 2000


def test_invalid_limit_and_missing_queries_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        module.select_queries({"a": ["x"]}, ["a"], max_queries=-1)
    with pytest.raises(ValueError, match="No caption"):
        module.select_queries({"a": ["x"]}, ["b"])
