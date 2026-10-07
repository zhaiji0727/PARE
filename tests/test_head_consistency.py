import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from saqre_phys import SAQREHead


def documents():
    torch.manual_seed(7)
    batch, dim = 3, 16
    docs = {"g": F.normalize(torch.randn(batch, dim), dim=-1),
            "agg": torch.randn(batch, dim), "nu": torch.rand(batch) * 0.1}
    for prefix, count in (("c", 4), ("f", 5)):
        docs[prefix + "_emb"] = F.normalize(torch.randn(batch, count, dim), dim=-1)
        docs[prefix + "_mu"] = torch.rand(batch, count)
        docs[prefix + "_t"] = torch.rand(batch, count)
        docs[prefix + "_tf"] = torch.rand(batch, count, 4)
        docs[prefix + "_mask"] = torch.arange(count)[None, :] < torch.tensor([0, 2, count])[:, None]
    return docs


class HeadTests(unittest.TestCase):
    def test_single_and_batched_paths_match_including_empty_primitives(self):
        docs = documents()
        head = SAQREHead(16)
        query = F.normalize(torch.randn(16), dim=-1)
        for ablation in (set(), {"sr"}, {"ty"}, {"cfar"}, {"sr", "ty", "cfar"},
                         {"cfar_strict"}, {"cfar_strict", "sr"}, {"cfar_strict", "ty"},
                         {"cfar_strict", "sr", "ty"}):
            head.ablate = ablation
            expected = []
            for row in range(3):
                doc = {key: docs[key][row] for key in ("g", "agg", "nu")}
                for prefix in ("c", "f"):
                    mask = docs[prefix + "_mask"][row]
                    for suffix in ("emb", "mu", "t", "tf"):
                        key = prefix + "_" + suffix
                        doc[key] = docs[key][row][mask]
                expected.append(head(query, doc))
            torch.testing.assert_close(head.fuse_batch(query, docs), torch.stack(expected), atol=2e-6, rtol=2e-6)

    def test_strict_cfar_ignores_every_detector_information_path(self):
        docs = documents()
        altered = {key: value.clone() for key, value in docs.items()}
        for suffix in ("emb", "mu", "t", "tf"):
            altered["f_" + suffix].fill_(float("nan"))
        altered["f_mask"] = ~altered["f_mask"]
        altered["agg"].fill_(float("nan"))
        altered["c_tf"][..., 3] = float("nan")
        head = SAQREHead(16)
        query = F.normalize(torch.randn(16), dim=-1)
        for flags in ({"cfar_strict"}, {"cfar_strict", "sr"}, {"cfar_strict", "ty"},
                      {"cfar_strict", "sr", "ty"}):
            head.ablate = flags
            expected = head.fuse_batch(query, docs)
            actual = head.fuse_batch(query, altered)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual).all())
            for row in range(3):
                single = {key: altered[key][row] for key in ("g", "agg", "nu")}
                for prefix in ("c", "f"):
                    mask = altered[prefix + "_mask"][row]
                    for suffix in ("emb", "mu", "t", "tf"):
                        key = prefix + "_" + suffix
                        single[key] = altered[key][row][mask]
                torch.testing.assert_close(head(query, single), expected[row], atol=2e-6, rtol=2e-6)

    def test_legacy_cfar_retains_aggregate_and_descriptor_paths(self):
        docs = documents()
        head = SAQREHead(16)
        head.ablate = {"cfar"}
        query = F.normalize(torch.randn(16), dim=-1)
        original = head.fuse_batch(query, docs)
        changed_aggregate = {key: value.clone() for key, value in docs.items()}
        changed_aggregate["agg"] = -10 * changed_aggregate["agg"]
        self.assertFalse(torch.allclose(original, head.fuse_batch(query, changed_aggregate)))
        changed_descriptor = {key: value.clone() for key, value in docs.items()}
        changed_descriptor["c_tf"][..., 3] += 20
        self.assertFalse(torch.allclose(original, head.fuse_batch(query, changed_descriptor)))

    def test_strict_cfar_has_finite_training_gradients(self):
        docs = documents()
        head = SAQREHead(16)
        head.ablate = {"cfar_strict"}
        queries = F.normalize(torch.randn(3, 16), dim=-1)
        scores = torch.stack([head.fuse_batch(query, docs) @ query for query in queries])
        F.cross_entropy(scores / 0.03, torch.arange(3)).backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_empty_primitive_rows_have_finite_training_gradients(self):
        docs = documents()
        head = SAQREHead(16)
        queries = F.normalize(torch.randn(3, 16), dim=-1)
        scores = torch.stack([head.fuse_batch(query, docs) @ query for query in queries])
        loss = F.cross_entropy(scores / 0.03, torch.arange(3))
        loss.backward()
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_soft_reliability_preserves_low_return_regions(self):
        head = SAQREHead(16)
        values = head.reliability(torch.tensor([0.0, 0.5]), torch.tensor([0.0, 1.0]), torch.tensor(0.05)).detach()
        self.assertGreater(float(values[0]), 0.0)
        self.assertGreater(float(values[1]), float(values[0]))
        self.assertLessEqual(float(values.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
