import importlib.util
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clean_split", ROOT / "scripts/prepare_content_disjoint_split.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pixel_hash_ignores_encoding_but_includes_dimensions(tmp_path):
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    png, tiff, other = [tmp_path / name for name in ("a.png", "b.tif", "c.png")]
    image.save(png)
    image.save(tiff)
    Image.new("RGB", (3, 4), (10, 20, 30)).save(other)
    a, b, c = [module.image_fingerprint(path) for path in (png, tiff, other)]
    assert a["file_sha256"] != b["file_sha256"]
    assert a["rgb_content_sha256"] == b["rgb_content_sha256"]
    assert a["rgb_content_sha256"] != c["rgb_content_sha256"]


def test_test_priority_canonical_train_and_deterministic_split():
    train = ["z", "b", "a", "c", "d", "e"]
    test = ["t", "t2"]
    keys = dict(z="T", b="A", a="A", c="C", d="D", e="E", t="T", t2="T")
    records = [{"image_id": key, "rgb_content_sha256": value} for key, value in keys.items()]
    result = module.make_protocol(train, test, records, val_fraction=0.25)
    assert result == module.make_protocol(train, test, records, val_fraction=0.25)
    assert result["splits"]["train"] == ["a", "c", "d", "e"]
    assert result["splits"]["test"] == test
    assert len(result["splits"]["val"]) == 1
    assert not set(result["splits"]["fit"]) & set(result["splits"]["val"])
    assert {row["image_id"]: row["reason"] for row in result["excluded_training_ids"]} == {
        "z": "content_already_in_official_test", "b": "noncanonical_training_content_duplicate"}
    assert result["official_test_duplicate_content_groups"] == [{"rgb_content_sha256": "T", "image_ids": test}]


def test_caption_filter_retains_only_canonical_rows_and_source(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("Image Name, Caption\na,alpha\nb,beta\na,second\n")
    before = source.read_bytes()
    dest = tmp_path / "filtered.csv"
    result = module.write_filtered_captions(source, dest, ["a"])
    assert result == {"rows": 2, "nonempty_rows": 2, "unique_captioned_images": 1}
    assert "beta" not in dest.read_text()
    assert source.read_bytes() == before


def test_missing_fingerprints_rejected():
    with pytest.raises(ValueError, match="cover exactly"):
        module.make_protocol(["a"], ["b"], [])
