import contextlib
import io
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Automated_FA"))

from Lib import api_utils  # noqa: E402


class FormalBaselineSmokeTest(unittest.TestCase):
    def setUp(self):
        dataset_root = Path(
            os.environ.get(
                "WHO_WHEN_ROOT",
                REPO_ROOT / "data" / "Agents_Failure_Attribution" / "Who&When",
            )
        )
        self.sample = dataset_root / "Algorithm-Generated" / "1.json"
        if not self.sample.exists():
            self.skipTest(
                "Who&When is unavailable. Clone the official benchmark as documented in README.md."
            )

    def test_all_formal_baselines_run_on_official_sample(self):
        responses = {
            "all_at_once": "Agent Name: Excel_Expert\nStep Number: 0\nReason for Mistake: test",
            "step_by_step": "1. Yes.\n2. Reason: decisive error",
            "binary_search": "upper half",
        }
        methods = {
            "all_at_once": api_utils.all_at_once_api,
            "step_by_step": api_utils.step_by_step_api,
            "binary_search": api_utils.binary_search_api,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "alg"
            input_dir.mkdir()
            shutil.copy2(self.sample, input_dir / self.sample.name)

            for name, method in methods.items():
                with self.subTest(method=name):
                    output = io.StringIO()
                    with (
                        mock.patch.object(
                            api_utils,
                            "_make_api_call",
                            return_value=responses[name],
                        ),
                        contextlib.redirect_stdout(output),
                    ):
                        method(
                            client=object(),
                            directory_path=str(input_dir),
                            is_handcrafted=False,
                            model="deepseek-ai/DeepSeek-V3.2",
                            max_tokens=256,
                        )
                    self.assertIn("1.json", output.getvalue())

    def test_selected_model_is_forwarded_to_openai_client(self):
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                delta = SimpleNamespace(content="ok")
                choice = SimpleNamespace(delta=delta)
                return [SimpleNamespace(choices=[choice])]

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        result = api_utils._make_api_call(
            client=client,
            model="deepseek-ai/DeepSeek-V3.2",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=16,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["model"], "deepseek-ai/DeepSeek-V3.2")


if __name__ == "__main__":
    unittest.main()
