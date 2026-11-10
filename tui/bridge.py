#!/usr/bin/env python3
"""JSON-lines adapter for the interactive hegemon CLI.

Only this process writes protocol events to stdout. Prompt replies are read from
stdin as {"type": "answer", "value": "..."}; they are never echoed.
"""
import getpass
import json
import sys
import threading
import traceback
from pathlib import Path


# Running this file by path puts tui/, rather than the project root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Protocol:
    def __init__(self, output, input_stream):
        self.output = output
        self.input = input_stream
        self.lock = threading.RLock()
        self.prompt_lock = threading.Lock()
        self.finished = False

    def emit(self, event):
        with self.lock:
            if self.finished:
                return
            self.output.write(json.dumps(event, ensure_ascii=True) + "\n")
            self.output.flush()
            if event.get("type") == "exit":
                self.finished = True

    def prompt(self, text, default=None, secret=False):
        # Serialize prompts as well as event writes, so replies cannot go to
        # the wrong caller if another thread ever asks a question.
        with self.prompt_lock:
            event = {"type": "prompt", "prompt": text}
            if default is not None:
                event["default"] = default
            if secret:
                event["secret"] = True
            self.emit(event)
            line = self.input.readline()
            if not line:
                raise EOFError("Input closed while waiting for an answer.")
            try:
                answer = json.loads(line)
            except ValueError:
                raise ValueError("Expected a JSON answer event.") from None
            if not isinstance(answer, dict) or answer.get("type") != "answer" or not isinstance(answer.get("value"), str):
                raise ValueError("Expected an answer event with a string value.")
            return answer["value"]


class OutputStream:
    """Translate each write exactly once, including writes without newlines."""
    def __init__(self, protocol):
        self.protocol = protocol
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        if text:
            self.protocol.emit({"type": "output", "text": text})
        return len(text)

    def flush(self):
        pass  # emit flushes every event

    def isatty(self):
        return False  # Do not put terminal escape sequences into output events.

    def writable(self):
        return True


class InteractiveInput:
    """Retain the real input stream but advertise an interactive session."""
    def __init__(self, stream):
        self.stream = stream

    def isatty(self):
        return True

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main(argv=None):
    protocol = Protocol(sys.stdout, sys.stdin)
    sys.stdout = sys.stderr = OutputStream(protocol)
    sys.stdin = InteractiveInput(protocol.input)
    code = 1
    try:
        from hegemon_nb import ui, llm
        import hegemon

        def ask(prompt, default=""):
            answer = protocol.prompt(prompt, default=default).strip()
            return answer or default

        def yes_no(prompt, default=False, assume=None):
            if assume is not None:
                ui.info("{} {}".format(prompt, "yes (--yes)" if assume else "no"))
                return assume
            suffix = " [Y/n] " if default else " [y/N] "
            answer = protocol.prompt(prompt + suffix, default="y" if default else "n").strip().lower()
            return answer.startswith("y") if answer else default

        def secret_prompt(prompt="Password: ", stream=None):
            return protocol.prompt(prompt, secret=True)

        ui.ask = ask
        ui.yes_no = yes_no
        # llm imported the function directly; cmd_key imports it at call time.
        llm.getpass = secret_prompt
        getpass.getpass = secret_prompt
        result = hegemon.main(sys.argv[1:] if argv is None else argv)
        code = 0 if result is None else int(result)
    except SystemExit as exc:
        if exc.code is None:
            code = 0
        elif isinstance(exc.code, int):
            code = exc.code
        else:
            print(exc.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        print("\n   Stopped with Ctrl+C. Partial files stay in the build folder.", file=sys.stderr)
        code = 130
    except EOFError:
        # In particular, do not retry forever in setup_local after disconnect.
        print("Input closed while waiting for an answer.", file=sys.stderr)
        code = 1
    except Exception:
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        # OS exit statuses are eight-bit values on POSIX; report what the parent sees.
        code = code % 256
        protocol.emit({"type": "exit", "code": code})
    return code


if __name__ == "__main__":
    sys.exit(main())
