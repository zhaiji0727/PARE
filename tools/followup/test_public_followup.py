import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_public_followup import MANIFEST, verify

ROOT = Path(__file__).resolve().parents[2]


class PublicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = json.loads((ROOT / MANIFEST).read_text())
        for item in self.manifest['files']:
            src = ROOT / item['path']; dst = self.root / item['path']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        self.save_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        target = self.root / MANIFEST
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.manifest))

    def replace_summary(self, update):
        item = self.manifest['summary']
        path = self.root / item['path']
        value = json.loads(path.read_text()); update(value)
        path.write_text(json.dumps(value)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        item['sha256'] = digest
        for row in self.manifest['files']:
            if row['path'] == item['path']:
                row['sha256'] = digest
        self.save_manifest()

    def test_complete_public_package(self):
        self.assertTrue(verify(self.root)['all_passed'])

    def test_reject_changed_head(self):
        path = self.root / self.manifest['runs'][0]['head']['path']
        path.write_bytes(path.read_bytes() + b'corruption')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify(self.root)

    def test_reject_omitted_unfavorable_contrast_even_with_updated_hash(self):
        self.replace_summary(lambda s: s['tables']['contrast_summary'].pop(1))
        with self.assertRaisesRegex(ValueError, 'Length mismatch'):
            verify(self.root)

    def test_reject_changed_metric_even_with_updated_hash(self):
        self.replace_summary(lambda s: s['tables']['run_metrics'][0].update({'R@10': 100.0}))
        with self.assertRaisesRegex(ValueError, 'Numeric mismatch'):
            verify(self.root)

    def test_reject_duplicate_run(self):
        self.manifest['runs'][-1] = copy.deepcopy(self.manifest['runs'][0]); self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            verify(self.root)

    def test_reject_path_escape(self):
        self.manifest['runs'][0]['record']['path'] = '../outside.json.gz'; self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Unsafe public path'):
            verify(self.root)


if __name__ == '__main__':
    unittest.main()
