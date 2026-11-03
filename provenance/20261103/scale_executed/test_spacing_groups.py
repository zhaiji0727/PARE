from copy import deepcopy
import unittest

from spacing_groups import make_groups, spacing_bin, summarize_rank_rows


class SpacingGroupTests(unittest.TestCase):
    def test_rounding_stabilizes_nominal_boundary(self):
        for value in (0.299998, 0.3, 0.300007):
            self.assertEqual(spacing_bin({"raster_spacing_m_xy": [value, value]}), ("0.30_to_lt_0.45", "0.30"))
        self.assertEqual(spacing_bin({"raster_spacing_m_xy": [0.250003, 0.250003]}), ("lt_0.30", "0.25"))
        self.assertEqual(spacing_bin({"raster_spacing_m_xy": [0.45, 0.45]}), ("ge_0.45", "0.45"))
        self.assertEqual(spacing_bin({"raster_spacing_m_xy": [0.49997538676407427, 0.4999753760206335]}), ("ge_0.45", "0.50"))
        with self.assertRaises(ValueError):
            spacing_bin({"raster_spacing_m_xy": [0.2, 0.3]})

    def test_subgroup_keeps_global_rank_instead_of_recomputing_reduced_gallery(self):
        ids = [f"image_{i}" for i in range(12)]
        audit = {"images": {name: {"raster_spacing_m_xy": [0.2, 0.2] if i == 0 else None}
                             for i, name in enumerate(ids)},
                 "summaries": {"test": {"images": 12, "with_projected_metre_raster_spacing": 1}}}
        groups = make_groups(audit, ids)
        rows = [{"image_id": name, "ranks": {"PARE": 12, "Scene-CLIP": 1}} for name in ids]
        summary = summarize_rank_rows(rows, groups)
        self.assertEqual(summary["gallery_size"], 12)
        self.assertEqual(summary["subgroups"]["lt_0.30"]["n_queries"], 1)
        self.assertEqual(summary["subgroups"]["lt_0.30"]["metrics"]["PARE"]["R@10"], 0)
        self.assertEqual(summary["subgroups"]["lt_0.30"]["metrics"]["Scene-CLIP"]["R@1"], 100)
        with self.assertRaises(ValueError):
            summarize_rank_rows(rows[:1], groups)
        duplicated = deepcopy(rows)
        duplicated[-1] = duplicated[0]
        with self.assertRaises(ValueError):
            summarize_rank_rows(duplicated, groups)


if __name__ == "__main__":
    unittest.main(verbosity=2)
