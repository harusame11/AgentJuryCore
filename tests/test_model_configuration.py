import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Automated_FA"))

import inference  # noqa: E402
from Lib import api_utils  # noqa: E402


class ModelConfigurationTest(unittest.TestCase):
    def test_main_experiment_model_is_deepseek_v32(self):
        self.assertEqual(inference.MAIN_EXPERIMENT_MODEL, "ds-v3.2")
        self.assertEqual(
            api_utils.API_MODEL_MAP[inference.MAIN_EXPERIMENT_MODEL],
            "deepseek-ai/DeepSeek-V3.2",
        )

    def test_dao_calls_enable_deepseek_thinking(self):
        captured = []

        class Completions:
            def create(self, **kwargs):
                captured.append(kwargs)
                delta = SimpleNamespace(content="ok")
                return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        messages = [{"role": "user", "content": "test"}]

        api_utils._make_api_DAO_1_call(client, "ignored", messages)
        api_utils._make_api_DAO_2_call(client, "ignored", messages)
        api_utils._make_api_DAO_3_call(client, "ignored", messages)

        self.assertEqual(len(captured), 3)
        for request in captured:
            self.assertEqual(request["model"], "deepseek-ai/DeepSeek-V3.2")
            self.assertEqual(
                request["extra_body"],
                {"thinking": {"type": "enabled"}},
            )

    def test_algorithm_probes_use_deepseek_thinking(self):
        source = inspect.getsource(api_utils.AgentJury_alg_enhance)
        source += inspect.getsource(api_utils.AgentJury_alg_enhance_noGT)
        self.assertNotIn('"kimi-2.5"', source)
        self.assertIn('"ds-v3.2"', source)
        self.assertIn("thinking=True", source)

    def test_semantic_mapping_uses_qwen35_35b(self):
        module_path = REPO_ROOT / "extract_of_error" / "extract_D.py"
        spec = importlib.util.spec_from_file_location("extract_D_config", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MODEL_NAME, "Qwen/Qwen3.5-35B-A3B")


if __name__ == "__main__":
    unittest.main()
