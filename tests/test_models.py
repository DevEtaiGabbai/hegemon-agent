"""Regression tests for model defaults and OpenAI Responses requests."""
import json
import os
import unittest
from unittest.mock import patch

from hegemon_nb.config import Config
from hegemon_nb.llm import LLM, ProviderError


RESPONSE = {
    "status": "completed",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
    "usage": {"input_tokens": 2, "output_tokens": 1},
}


class ModelTests(unittest.TestCase):
    def config(self, **env):
        # Avoid inheriting model, mode, or saved-settings overrides from the host.
        with patch.dict(os.environ, env, clear=True), patch("hegemon_nb.config.load_settings", return_value={}):
            return Config()

    def test_defaults_and_mode_routing(self):
        cloud = self.config()
        self.assertEqual(cloud.model, "gpt-6.1-sol")
        self.assertEqual(cloud.fast_model, "gpt-6.1-sol")
        mixed = self.config(HEGEMON_MODE="mixed", HEGEMON_LOCAL_MODEL="local-test")
        self.assertEqual((mixed.model, mixed.fast_model), ("gpt-6.1-sol", "local-test"))
        local = self.config(HEGEMON_MODE="local", HEGEMON_LOCAL_MODEL="local-test")
        self.assertEqual((local.model, local.fast_model), ("local-test", "local-test"))

    def test_cloud_overrides_survive_mode_changes(self):
        cfg = self.config(HEGEMON_MODEL="custom-main", HEGEMON_FAST_MODEL="custom-fast",
                          HEGEMON_MODE="local", HEGEMON_LOCAL_MODEL="local-test")
        self.assertEqual((cfg.model, cfg.fast_model), ("local-test", "local-test"))
        cfg.mode = "cloud"
        cfg.apply_mode()
        self.assertEqual((cfg.model, cfg.fast_model), ("custom-main", "custom-fast"))

    def test_responses_payload_and_per_call_override(self):
        cfg = self.config()
        llm = LLM("key", cfg)
        with patch.object(llm, "_post", return_value=RESPONSE) as post:
            self.assertEqual(llm.ask("system", "prompt"), "OK")
            self.assertEqual(post.call_args.args[0], {
                "model": "gpt-6.1-sol", "instructions": "system", "input": "prompt",
                "max_output_tokens": 32000,
            })
            llm.ask("system", "prompt", model="other-model", effort="high")
            self.assertEqual(post.call_args.args[0]["model"], "other-model")
            self.assertEqual(post.call_args.args[0]["reasoning"], {"effort": "high"})

    def test_responses_http_request(self):
        llm = LLM("test-key", self.config())
        with patch("hegemon_nb.llm.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(RESPONSE).encode()
            self.assertEqual(llm.ask("system", "prompt"), "OK")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(request.data)["model"], "gpt-6.1-sol")

    def test_reasoning_rejection_retries_without_reasoning(self):
        cfg = self.config(HEGEMON_REASONING="high")
        llm = LLM("key", cfg)
        payloads = []

        def post(payload, timeout):
            payloads.append(dict(payload))
            if len(payloads) == 1:
                raise ProviderError("bad_request", "reasoning is unsupported")
            return RESPONSE

        with patch.object(llm, "_post", side_effect=post):
            self.assertEqual(llm.ask("system", "prompt"), "OK")
        self.assertEqual(payloads[0]["reasoning"], {"effort": "high"})
        self.assertNotIn("reasoning", payloads[1])
        self.assertEqual(payloads[1]["model"], "gpt-6.1-sol")


if __name__ == "__main__":
    unittest.main()
