"""Temporary synthetic fixtures test layout gates and table membership."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import export_paper_results as exporter
import layout_paper_tables as layout
from test_export_paper_results import make_fixture, make_stress


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        evaluation_dir, sweep_path, _ = make_fixture(cls.root)
        evaluation = exporter.load_evaluation("vitb32", evaluation_dir)
        sweep, _ = exporter.load_sweep("vitb32", sweep_path, evaluation)
        stress_dir, _ = make_stress(cls.root, evaluation)
        stress = exporter.load_stress("vitb32", stress_dir, evaluation)
        cls.tables = exporter.build_tables({tag: evaluation for tag in exporter.TAGS},
                                          {tag: sweep for tag in exporter.TAGS}, {"vitb32": stress})
        cls.export = cls.root / "verified_export"
        provenance = {"schema": "pare.paper-numeric-export.v1", "source_sha256": {"export_paper_results.py": layout.EXPORTER_SHA},
                      "evaluations": dict.fromkeys(layout.TAGS, {}), "sweeps": dict.fromkeys(layout.TAGS, {}),
                      "stresses": {"vitb32": {}}, "test_fixture_only": True}
        exporter.write_exports(cls.export, cls.tables, provenance)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_json_and_csv_entry_points_agree(self):
        a = layout.load_verified_export(self.export / "paper_results.json")[2]
        b = layout.load_verified_export(self.export / "table_i_full_gallery.csv")[2]
        self.assertEqual(a, b)
        result = layout.render(self.export, self.root / "layout")
        self.assertTrue(result["main_table_iii_included"])
        self.assertEqual(result["table_row_counts"]["appendix_natural_k"], 120)
        self.assertEqual(result["table_row_counts"]["appendix_validation_grid"], 45)
        self.assertIn("supp_scale_vitb32.tex", result["outputs"])
        self.assertEqual(result["main_table_i_display_methods"][-1], "PARE-CFAR-strict")
        self.assertEqual(result["main_table_i_added_row_source"]["table"], "appendix_natural_k")
        self.assertEqual(result["main_table_i_added_row_source"]["K"], 3688)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            layout.render(self.export, self.root / "layout")

    def test_main_columns_and_all_six_methods(self):
        files = layout.main_tables(self.tables)
        first = files["main_table_i_tabular.tex"]
        data = first.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0].strip().splitlines()
        self.assertEqual(len(data), 6)
        self.assertTrue(all(row.count(" & ") == 20 for row in data))
        self.assertTrue(data[0].startswith("Scene-CLIP"))
        self.assertTrue(data[3].startswith("Stats-off"))
        self.assertTrue(data[4].startswith("PARE"))
        self.assertTrue(data[5].startswith("CFAR-strict"))
        second = files["main_table_ii_tabular.tex"].split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0].strip().splitlines()
        self.assertEqual(len(second), 4)
        self.assertTrue(all(row.count(" & ") == 5 for row in second))
        third = files["main_table_iii_tabular.tex"].split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0].strip().splitlines()
        self.assertEqual(len(third), 3)
        self.assertTrue(all(row.count(" & ") == 4 for row in third))
        self.assertIn("-50.00", files["main_table_iii_tabular.tex"])
        self.assertNotIn("textbf", "".join(files.values()))
        self.assertIn(r"\footnotesize", first)
        self.assertIn(r"\ifdim\wd0>\textwidth", first)
        self.assertIn(r"\else\box0\fi", first)
        self.assertIn(r"\ifdim\wd0>\linewidth", files["main_table_ii_tabular.tex"])

    def test_added_strict_row_uses_only_complete_gallery_and_preserves_export(self):
        tables = copy.deepcopy(self.tables)
        expected = []
        for index, tag in enumerate(layout.TAGS):
            values = [index + 1.0, index + 2.0, index + 3.0, index + 1.5]
            expected.extend(layout.number(value) for value in values)
            for row in tables["appendix_natural_k"]:
                if row["backbone"] == tag and row["method"] == "PARE-CFAR-strict":
                    row.update(zip(layout.METRICS, values if row["K"] == 3688 else [91.0, 92.0, 93.0, 91.5]))
        original = copy.deepcopy(tables)
        files = layout.main_tables(tables)
        strict_line = next(line for line in files["main_table_i_tabular.tex"].splitlines() if line.startswith("CFAR-strict &"))
        self.assertEqual(strict_line, layout.line(["CFAR-strict"] + expected))
        self.assertEqual(tables, original)
        self.assertEqual(len(tables["table_i_full_gallery"]), 25)
        original_files = layout.main_tables(self.tables)
        for name in original_files:
            if not name.startswith("main_table_i_"):
                self.assertEqual(files[name], original_files[name])

    def test_no_unmeasured_scale_table(self):
        tables = {key: value for key, value in self.tables.items() if key != "synthetic_scale_sensitivity"}
        layout.validate_tables(tables, set())
        files = layout.main_tables(tables)
        self.assertNotIn("main_table_iii_tabular.tex", files)
        self.assertFalse(any("supp_scale" in name for name in layout.supplementary_tables(tables)))

    def test_supplement_preserves_all_rows_and_metrics(self):
        files = layout.supplementary_tables(self.tables)
        self.assertEqual(len([name for name in files if name.startswith("supp_natural_")]), 5)
        for name, content in files.items():
            data = content.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0].strip().splitlines()
            data = [row for row in data if not row.startswith("\\addlinespace")]
            if name.startswith("supp_natural_"):
                self.assertEqual(len(data), 24)
                self.assertTrue(all(row.count(" & ") == 11 for row in data))
                self.assertEqual(content.count("MRR@10"), 2)
            if name.startswith(("supp_controlled100_", "supp_hard50_")):
                self.assertEqual(len(data), 7)
                self.assertIn("not end-to-end", content)
            if name.startswith("supp_validation_"):
                self.assertEqual(len(data), 9)
                self.assertEqual(sum("Yes" in row for row in data), 1)
            if name.startswith("supp_scale_"):
                self.assertEqual(len(data), 9)
                self.assertIn("R@5", content)

    def test_reject_incomplete_or_inconsistent_tables(self):
        tables = copy.deepcopy(self.tables)
        tables["table_i_full_gallery"].pop()
        with self.assertRaisesRegex(ValueError, "Missing"):
            layout.validate_tables(tables, {"vitb32"})
        tables = copy.deepcopy(self.tables)
        tables["table_ii_vitb32_controls"][0]["natural100_R@10"] += 1
        with self.assertRaisesRegex(ValueError, "Table II"):
            layout.validate_tables(tables, {"vitb32"})
        tables = copy.deepcopy(self.tables)
        tables["synthetic_scale_sensitivity"][-1]["delta_R10_pp"] *= -1
        with self.assertRaisesRegex(ValueError, "Scale delta"):
            layout.validate_tables(tables, {"vitb32"})

    def test_noncomplete_hash_and_version_gates(self):
        path = self.export / "export_manifest.json"
        original = path.read_text()
        try:
            manifest = json.loads(original)
            manifest["status"] = "partial"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "completed"):
                layout.load_verified_export(self.export)
            manifest["status"] = "complete"
            manifest["source_sha256"]["export_paper_results.py"] = "not-reviewed"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "reviewed"):
                layout.load_verified_export(self.export)
            manifest = json.loads(original)
            manifest["outputs"]["paper_results.json"]["sha256"] = "wrong"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash"):
                layout.load_verified_export(self.export)
        finally:
            path.write_text(original)

    def test_csv_json_disagreement_rejected_even_if_csv_hash_is_updated(self):
        path = self.export / "table_i_full_gallery.csv"
        manifest_path = self.export / "export_manifest.json"
        original_csv = path.read_bytes()
        original_manifest = manifest_path.read_text()
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["R@10"] = "99.0"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            manifest = json.loads(original_manifest)
            manifest["outputs"][path.name] = {"sha256": layout.sha(path), "bytes": path.stat().st_size}
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "CSV/JSON value"):
                layout.load_verified_export(self.export)
        finally:
            path.write_bytes(original_csv)
            manifest_path.write_text(original_manifest)


if __name__ == "__main__":
    unittest.main()
