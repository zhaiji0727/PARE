import math
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_training_pairs import audit_batch, build_pairs, simulate


def test_pair_building_keeps_trainer_group_order_and_exact_duplicates():
    captions = {"b": ["second", "second"], "unused": ["skip"], "a": ["first", "other"]}
    assert build_pairs(captions, ["a", "b"]) == [
        ("b", "second"), ("b", "second"), ("a", "first"), ("a", "other")]


def test_counts_distinguish_different_and_identical_captions():
    result = audit_batch([("a", "one"), ("a", "two"), ("a", "two"), ("b", "three")])
    assert result["ordered_off_diagonal_candidates"] == 12
    assert result["same_image_off_diagonal_candidates"] == 6
    assert result["same_image_different_caption_candidates"] == 4
    assert result["same_image_identical_caption_candidates"] == 2
    assert result["queries_with_same_image_off_diagonal"] == 3
    assert result["queries_with_different_caption_same_image_off_diagonal"] == 3
    assert result["mean_query_ce_lower_bound_nats"] == pytest.approx(3 * math.log(3) / 4)
    assert result["histograms"]["query_image_multiplicity"] == {3: 3, 1: 1}
    assert result["histograms"]["query_exact_pair_multiplicity"] == {1: 2, 2: 2}


def test_exact_duplicates_are_same_image_but_not_different_caption():
    result = audit_batch([("a", "one"), ("a", "one")])
    assert result["same_image_off_diagonal_candidates"] == 2
    assert result["same_image_different_caption_candidates"] == 0
    assert result["queries_with_different_caption_same_image_off_diagonal"] == 0
    assert result["mean_query_ce_lower_bound_nats"] == pytest.approx(math.log(2))


def test_continuous_local_shuffle_and_retained_partial_batch():
    pairs = [("a", str(index)) for index in range(3)] + [("b", "x"), ("c", "y")]
    state = random.getstate()
    actual = simulate(pairs, batch_size=2, epochs=3, seed=0)
    assert random.getstate() == state
    assert actual == simulate(pairs, batch_size=2, epochs=3, seed=0)
    assert actual["aggregate"]["queries"] == 15
    assert actual["aggregate"]["batches"] == 9
    assert actual["aggregate"]["ordered_off_diagonal_candidates"] == 12
    assert [epoch["last_batch_size"] for epoch in actual["epochs"]] == [1, 1, 1]
    rng = random.Random(0)
    order = list(pairs)
    expected_collisions = 0
    for _ in range(3):
        rng.shuffle(order)
        for start in range(0, len(order), 2):
            batch = order[start:start + 2]
            expected_collisions += sum(a != b and batch[a][0] == batch[b][0]
                                       for a in range(len(batch)) for b in range(len(batch)))
    assert actual["aggregate"]["same_image_off_diagonal_candidates"] == expected_collisions


def test_singleton_has_zero_negative_denominator_and_bound():
    result = audit_batch([("a", "one")])
    assert result["ordered_off_diagonal_candidates"] == 0
    assert result["same_image_off_diagonal_fraction"] == 0
    assert result["mean_query_ce_lower_bound_nats"] == 0


def test_same_document_grouped_positive_ce_has_identical_shared_parameter_gradient():
    unique_image_logits = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
    image_ids = torch.tensor([0, 0, 0, 1, 1])
    logits = unique_image_logits[image_ids]
    diagonal_loss = torch.logsumexp(logits, dim=0) - logits[0]
    grouped_loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[image_ids == 0], dim=0)
    assert (diagonal_loss - grouped_loss).item() == pytest.approx(math.log(3))
    diagonal_gradient = torch.autograd.grad(diagonal_loss, unique_image_logits, retain_graph=True)[0]
    grouped_gradient = torch.autograd.grad(grouped_loss, unique_image_logits, retain_graph=True)[0]
    torch.testing.assert_close(diagonal_gradient, grouped_gradient, rtol=0, atol=1e-14)
    # Removing duplicate candidates is a different change: denominator weights change.
    unique_loss = torch.logsumexp(unique_image_logits, dim=0) - unique_image_logits[0]
    unique_gradient = torch.autograd.grad(unique_loss, unique_image_logits)[0]
    assert not torch.allclose(diagonal_gradient, unique_gradient)
