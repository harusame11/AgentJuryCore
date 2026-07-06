import contextlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Automated_FA"))

from Lib import api_utils  # noqa: E402
from Lib.api_utils_4_handcrafted import AgentJury_hc_enhance_noGT  # noqa: E402


class HandCraftedNoGroundTruthSmokeTest(unittest.TestCase):
    def test_official_who_when_sample_runs_end_to_end(self):
        dataset_root = Path(
            os.environ.get(
                "WHO_WHEN_ROOT",
                REPO_ROOT / "data" / "Agents_Failure_Attribution" / "Who&When",
            )
        )
        sample = dataset_root / "Hand-Crafted" / "1.json"
        if not sample.exists():
            self.skipTest(
                "Who&When is unavailable. Clone the official benchmark as documented in README.md."
            )

        summary_response = (
            '{"Upstream_Instruction_or_Context":"test context",'
            '"Node_Action":"test action",'
            '"Result_and_Feedback":"test result"}'
        )
        probe_response = (
            '{"suspicious_steps":[{"step":3,"agent":"Orchestrator",'
            '"confidence":0.9,"reason":"issued the decisive instruction"}]}'
        )
        dao_response = (
            '<json>{"agent_evaluations":[{"agent_name":"Orchestrator",'
            '"step_index":3,"error_likelihood":0.9,'
            '"reasoning":"issued the decisive instruction"}]}</json>'
        )

        def fake_dao_call(client, model, messages, max_tokens=None):
            user_prompt = messages[-1]["content"]
            return dao_response if "agent_evaluations" in user_prompt else summary_response

        def fake_probe_call(client, model_name, messages, **kwargs):
            return probe_response

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "hc"
            input_dir.mkdir()
            shutil.copy2(sample, input_dir / sample.name)

            previous_cwd = Path.cwd()
            output = io.StringIO()
            try:
                os.chdir(temp_dir)
                with (
                    mock.patch.object(
                        api_utils,
                        "_make_api_call_with_retry",
                        side_effect=fake_probe_call,
                    ),
                    mock.patch.object(
                        api_utils,
                        "_make_api_DAO_1_call",
                        side_effect=fake_dao_call,
                    ),
                    mock.patch.object(
                        api_utils,
                        "_make_api_DAO_2_call",
                        side_effect=fake_dao_call,
                    ),
                    mock.patch.object(
                        api_utils,
                        "_make_api_DAO_3_call",
                        side_effect=fake_dao_call,
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    AgentJury_hc_enhance_noGT(
                        client=object(),
                        directory_path=str(input_dir),
                        model="deepseek-ai/DeepSeek-V3.2",
                        max_tokens=256,
                        target_seg_size=15,
                        max_probe_k=8,
                    )
            finally:
                os.chdir(previous_cwd)

        result = output.getvalue()
        self.assertIn("Prediction for 1.json: Error found.", result)
        self.assertIn("Agent Name: Orchestrator", result)
        self.assertIn("Step Number: 3", result)


if __name__ == "__main__":
    unittest.main()
