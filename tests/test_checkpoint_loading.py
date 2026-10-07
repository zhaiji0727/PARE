import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from runtime import load_checkpoint_strict


class CheckpointTests(unittest.TestCase):
    def test_complete_wrapped_checkpoint_loads(self):
        model = torch.nn.Linear(3, 2)
        state = {"module." + key: torch.ones_like(value) for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "complete.pt"
            torch.save({"state_dict": state}, path)
            load_checkpoint_strict(model, path)
        self.assertTrue(torch.equal(model.weight, torch.ones_like(model.weight)))

    def test_missing_keys_are_not_silently_accepted(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "incomplete.pt"
            torch.save({"weight": torch.ones_like(model.weight)}, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint_strict(model, path)


if __name__ == "__main__":
    unittest.main()
