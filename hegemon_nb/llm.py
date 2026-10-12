"""Minimal OpenAI Responses API client, plus a local-model route (standard library only).

Local models (Ollama, LM Studio, vLLM) are reached through their OpenAI-compatible chat-completions
endpoint at cfg.local_url. Which calls go local is decided by cfg.is_local(model) (see config.py).

- One request per call, with a hard timeout. The only retries are one visible
  retry for rate limits / server errors, printed to the terminal.
- Key, quota, model-access and timeout problems raise ProviderError, which the
  agent reports as an AI-provider problem, never as "dataset failed".
"""
from getpass import getpass
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request

from . import ui

API_URL = os.environ.get("HEGEMON_OPENAI_URL", "https://api.openai.com/v1/responses")


class ProviderError(Exception):
    def __init__(self, kind, message):
        Exception.__init__(self, message)
        self.kind = kind


def load_key(cfg, prompt=True):
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    if cfg.key_file.is_file():
        key = cfg.key_file.read_text().strip()
        if key:
            return key
    if not prompt:
        return None
    print("\n   OpenAI API key needed (input hidden). It is saved to {} (mode 600).".format(cfg.key_file))
    key = getpass("   OpenAI API key: ").strip()
    if key:
        save_key(cfg, key)
    return key or None


def save_key(cfg, key):
    cfg.key_file.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(str(cfg.key_file.parent), 0o700)
    tmp = cfg.key_file.with_suffix(".tmp")
    tmp.write_text(key.strip() + "\n")
    os.chmod(str(tmp), 0o600)
    os.replace(str(tmp), str(cfg.key_file))


def _error_from_http(code, body):
    try:
        err = json.loads(body).get("error") or {}
    except ValueError:
        err = {}
    msg = err.get("message") or body[:300] or "no details"
    etype = (err.get("code") or err.get("type") or "")
    if code == 401:
        return ProviderError("auth", "OpenAI rejected the API key (401). Run: hegemon key")
    if code == 429 and etype == "insufficient_quota":
        return ProviderError("quota", "OpenAI account is out of credit/quota (429 insufficient_quota): " + msg)
    if code == 429:
        return ProviderError("rate", "OpenAI rate limit (429): " + msg)
    if code in (403, 404) or "model" in etype:
        return ProviderError("model", "OpenAI refused the model ({}): {}. Set HEGEMON_MODEL to a model your key can use.".format(code, msg))
    if code >= 500:
        return ProviderError("server", "OpenAI server error {}: {}".format(code, msg))
    return ProviderError("bad_request", "OpenAI request rejected ({}): {}".format(code, msg))


LOCAL_MAX_OUTPUT = 8192   # a local model's window also has to hold the prompt


def _local_error(code, body, cfg, model):
    try:
        err = json.loads(body).get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except ValueError:
        msg = ""
    msg = msg or body[:300] or "no details"
    if code == 404 or "not found" in msg.lower():
        return ProviderError("model", "The local model {} was not found ({}). On the laptop run: ollama list "
                                      "(or: ollama pull {}), or change the name with: hegemon mode".format(model, msg, model))
    if code >= 500:
        return ProviderError("server", "The local model failed ({}): {}".format(code, msg))
    return ProviderError("bad_request", "The local model rejected the request ({}): {}".format(code, msg))


class LLM(object):
    def __init__(self, key, cfg):
        self.key = key
        self.cfg = cfg
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0      # input tokens OpenAI served from its prompt cache (billed at a discount)
        self.reasoning_tokens = 0   # part of output_tokens: hidden thinking, billed as output

    def _post(self, payload, timeout):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(API_URL, data=data, headers={
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            raise _error_from_http(err.code, err.read().decode("utf-8", "replace"))
        except socket.timeout:
            raise ProviderError("timeout", "No answer from OpenAI within {:.0f}s (HEGEMON_AI_TIMEOUT).".format(timeout))
        except urllib.error.URLError as err:
            if isinstance(getattr(err, "reason", None), socket.timeout):
                raise ProviderError("timeout", "No answer from OpenAI within {:.0f}s.".format(timeout))
            raise ProviderError("network", "Could not reach OpenAI: {}".format(err.reason))

    def _post_local(self, payload, timeout):
        url = self.cfg.local_url + "/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            raise _local_error(err.code, err.read().decode("utf-8", "replace"), self.cfg, payload.get("model"))
        except (socket.timeout, ValueError) as err:
            if isinstance(err, ValueError):
                raise ProviderError("network", "The local model at {} sent back something that is not JSON.".format(url))
            raise ProviderError("timeout", "No answer from the local model within {:.0f}s.".format(timeout))
        except (urllib.error.URLError, OSError) as err:
            reason = getattr(err, "reason", err)
            if isinstance(reason, socket.timeout):
                raise ProviderError("timeout", "No answer from the local model within {:.0f}s.".format(timeout))
            raise ProviderError("network", "Could not reach the local model at {} ({}). Is Ollama running on the "
                                           "laptop, and did you log in with: ssh -R 11434:127.0.0.1:11434 ...".format(
                                               self.cfg.local_url, reason))

    def _ask_local(self, instructions, text, model, max_output_tokens, timeout, label):
        payload = {"model": model, "stream": False, "max_tokens": min(max_output_tokens, LOCAL_MAX_OUTPUT),
                   "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": text}]}
        timeout = max(timeout or 0, self.cfg.ai_timeout)       # a laptop GPU is slower than OpenAI
        with ui.Heartbeat("waiting on {} (local: {})".format(label, model)) as hb:
            try:
                data = self._post_local(payload, timeout)
            except ProviderError as err:
                if err.kind != "server":
                    raise
                ui.warn(str(err) + " -- retrying once in 10s")
                time.sleep(10)
                data = self._post_local(payload, timeout)
        self.calls += 1
        usage = data.get("usage") or {}
        p_in, p_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        self.input_tokens += p_in
        self.output_tokens += p_out
        choice = (data.get("choices") or [{}])[0] or {}
        result = ((choice.get("message") or {}).get("content") or "")
        result = re.sub(r"(?s)<think>.*?</think>", "", result).strip()
        ui.info("{} answered in {:.0f}s ({} in / {} out tokens, local)".format(label, hb.elapsed, p_in, p_out))
        sent = (len(instructions) + len(text)) / 4.0       # rough token count of what was sent
        if p_in and sent > p_in * 1.3:
            ui.warn("the local model only read about {} of ~{:.0f} prompt tokens: its context window cut the "
                    "prompt short, so this answer is likely poor. Give the model a bigger num_ctx.".format(p_in, sent))
        if not result:
            raise ProviderError("empty", "The local model returned no text (finish reason: {}).".format(
                choice.get("finish_reason", "unknown")))
        return result

    def ask(self, instructions, text, model=None, effort=None, max_output_tokens=32000,
            timeout=None, label="AI", cache_key=None):
        """cache_key: calls that share the same instructions should share a key, so OpenAI routes them to
        the same prompt cache and bills the repeated prefix at the cached rate."""
        if self.cfg.is_local(model or self.cfg.model):
            return self._ask_local(instructions, text, model or self.cfg.model, max_output_tokens, timeout, label)
        payload = {
            "model": model or self.cfg.model,
            "instructions": instructions,
            "input": text,
            "max_output_tokens": max_output_tokens,
        }
        effort = self.cfg.reasoning if effort is None else effort
        if effort:
            payload["reasoning"] = {"effort": effort}
        if cache_key:
            payload["prompt_cache_key"] = cache_key
        timeout = timeout or self.cfg.ai_timeout
        retried = False
        while True:
            try:
                with ui.Heartbeat("waiting on {} ({})".format(label, payload["model"])) as hb:
                    data = self._post(payload, timeout)
                break
            except ProviderError as err:
                if err.kind == "bad_request" and "reasoning" in str(err) and "reasoning" in payload:
                    ui.warn("model does not take a reasoning setting; retrying without it")
                    payload.pop("reasoning")
                    continue
                if err.kind == "bad_request" and "prompt_cache_key" in str(err) and "prompt_cache_key" in payload:
                    ui.warn("endpoint does not take a prompt_cache_key; retrying without it")
                    payload.pop("prompt_cache_key")
                    continue
                if err.kind in ("rate", "server") and not retried:
                    retried = True
                    ui.warn(str(err) + " -- retrying once in 20s")
                    time.sleep(20)
                    continue
                raise
        self.calls += 1
        usage = data.get("usage") or {}
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        thinking = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
        self.cached_tokens += cached
        self.reasoning_tokens += thinking
        text_out = []
        for item in data.get("output") or []:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") in ("output_text", "text"):
                        text_out.append(part.get("text") or "")
        result = "".join(text_out).strip()
        ui.info("{} answered in {:.0f}s ({} in, {} cached / {} out, {} reasoning tokens)".format(
            label, hb.elapsed, usage.get("input_tokens", "?"), cached, usage.get("output_tokens", "?"), thinking))
        if data.get("status") == "incomplete" and not result:
            reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
            raise ProviderError("incomplete", "OpenAI stopped before answering (reason: {}).".format(reason))
        if not result:
            raise ProviderError("empty", "OpenAI returned no text.")
        return result

    def ping(self, model):
        """A real 1-request inference check (proves key, quota and model access)."""
        if self.cfg.is_local(model):
            data = self._post_local({"model": model, "stream": False, "max_tokens": 16,
                                     "messages": [{"role": "user", "content": "Reply with OK."}]}, timeout=300)
            if not (data.get("choices") or []):
                raise ProviderError("empty", "The local model answered with no reply.")
            return
        self._post({"model": model, "input": "Reply with OK.", "max_output_tokens": 64}, timeout=120)
