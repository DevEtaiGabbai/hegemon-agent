"""Subprocess-level regression tests for the TUI JSON-lines protocol."""
import json
import os
from pathlib import Path
import queue
import stat
import subprocess
import sys
import tempfile
import threading
import unittest


BRIDGE = Path(__file__).resolve().parent.parent / "tui" / "bridge.py"
TIMEOUT = 10


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        # Neither host settings nor environment overrides should affect the tests.
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("HEGEMON_")}
        self.env["HOME"] = self.home.name

    def run_bridge(self, *args, input_text=""):
        return subprocess.run(
            [sys.executable, str(BRIDGE), *args], input=input_text,
            text=True, capture_output=True, env=self.env, timeout=TIMEOUT,
        )

    def assert_protocol(self, stdout, stderr, returncode):
        self.assertEqual(stderr, "")
        events = [json.loads(line) for line in stdout.splitlines()]
        self.assertTrue(events, "bridge emitted no events")
        self.assertEqual(events[-1], {"type": "exit", "code": returncode})
        self.assertEqual(sum(event.get("type") == "exit" for event in events), 1)
        for event in events:
            self.assertIn(event.get("type"), ("output", "prompt", "exit"))
            if event["type"] == "output":
                self.assertIsInstance(event.get("text"), str)
        return events

    def answer_prompt(self, command, answer):
        """Wait for the prompt before sending a reply, with bounded waits."""
        proc = subprocess.Popen(
            [sys.executable, str(BRIDGE), command], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env,
        )
        lines = []
        prompts = queue.Queue()

        def read_output():
            for line in proc.stdout:
                lines.append(line)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # assert_protocol reports invalid JSON after exit
                if event.get("type") == "prompt":
                    prompts.put(event)
            prompts.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            prompt = prompts.get(timeout=TIMEOUT)
            self.assertIsNotNone(prompt, "bridge exited without prompting")
            proc.stdin.write(json.dumps({"type": "answer", "value": answer}) + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            returncode = proc.wait(timeout=TIMEOUT)
            reader.join(TIMEOUT)
            self.assertFalse(reader.is_alive(), "bridge output reader did not finish")
            stderr = proc.stderr.read()
            events = self.assert_protocol("".join(lines), stderr, returncode)
            self.assertEqual([e for e in events if e["type"] == "prompt"], [prompt])
            return prompt, events, returncode, "".join(lines), stderr
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            reader.join(TIMEOUT)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if not stream.closed:
                    stream.close()

    def test_help(self):
        result = self.run_bridge("--help")
        self.assertEqual(result.returncode, 0)
        events = self.assert_protocol(result.stdout, result.stderr, result.returncode)
        output = "".join(e["text"] for e in events if e["type"] == "output")
        self.assertIn("usage:", output.lower())
        self.assertIn("mode", output)
        self.assertFalse(any(e["type"] == "prompt" for e in events))

    def test_mode_prompt_roundtrip(self):
        prompt, events, code, _, _ = self.answer_prompt("mode", "1")
        self.assertEqual(code, 0)
        self.assertIn("Type 1, 2 or 3", prompt["prompt"])
        self.assertEqual(prompt.get("default"), "")
        self.assertNotIn("secret", prompt)
        settings = Path(self.home.name) / ".config" / "hegemon-agent" / "settings.json"
        self.assertEqual(json.loads(settings.read_text())["mode"], "cloud")
        self.assertTrue(any("saved:" in e.get("text", "") for e in events))

    def test_key_secret_prompt_and_storage(self):
        secret = "sk-test-bridge-secret-not-for-output"
        prompt, _, code, stdout, stderr = self.answer_prompt("key", secret)
        self.assertEqual(code, 0)
        self.assertIn("OpenAI API key", prompt["prompt"])
        self.assertIs(prompt.get("secret"), True)
        self.assertNotIn(secret, stdout)
        self.assertNotIn(secret, stderr)
        key_file = Path(self.home.name) / ".config" / "hegemon-agent" / "openai.key"
        self.assertEqual(key_file.read_text(), secret + "\n")
        self.assertEqual(stat.S_IMODE(key_file.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(key_file.parent.stat().st_mode), 0o700)

    def test_invalid_argument(self):
        result = self.run_bridge("--definitely-not-a-valid-option")
        self.assertEqual(result.returncode, 2)
        events = self.assert_protocol(result.stdout, result.stderr, result.returncode)
        output = "".join(e["text"] for e in events if e["type"] == "output")
        self.assertIn("error:", output.lower())
        self.assertFalse(any(e["type"] == "prompt" for e in events))

    def test_eof_while_waiting_for_answer(self):
        result = self.run_bridge("mode")
        self.assertEqual(result.returncode, 1)
        events = self.assert_protocol(result.stdout, result.stderr, result.returncode)
        prompts = [e for e in events if e["type"] == "prompt"]
        self.assertEqual(len(prompts), 1)
        self.assertIn("Type 1, 2 or 3", prompts[0]["prompt"])
        output = "".join(e["text"] for e in events if e["type"] == "output")
        self.assertIn("Input closed while waiting for an answer.", output)
        self.assertFalse((Path(self.home.name) / ".config" / "hegemon-agent" / "settings.json").exists())


if __name__ == "__main__":
    unittest.main()
