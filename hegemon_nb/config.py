"""All paths and limits in one place. Every value can be overridden with an env var."""
import getpass
import json
import os
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
BLUEPRINT_DIR = AGENT_DIR / "blueprints"
VERSION = "1.2"  # 1.2: local / mixed / cloud models (hegemon mode). 1.1: reruns add only what is new

# From the lab notebooks (both call this exact script for thr/info/vinfo/bv).
LAB_SCRIPT_CANDIDATES = (
    "/booleanfs2/sahoo/Data/BooleanLab/Training/GSE_Processing_scripts/jupyter_gse_processing",
    "/mnt/booleanfs2/sahoo/Data/BooleanLab/Training/GSE_Processing_scripts/jupyter_gse_processing",
)
# From the GSE notebook (Ensembl -> gene name tables).
GENOME_CANDIDATES = (
    "/booleanfs2/sahoo/Data/SeqData/genome",
    "/mnt/booleanfs2/sahoo/Data/SeqData/genome",
)


def _user():
    return os.environ.get("USER") or os.environ.get("USERNAME") or getpass.getuser()


def _first(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return Path(paths[0])


def _num(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


MODES = ("cloud", "local", "mixed")
MODE_TEXT = {"cloud": "cloud (OpenAI does everything)",
             "local": "local (your own model does everything, free)",
             "mixed": "mixed (your own model ranks and picks files, OpenAI writes the conversions)"}


def settings_file():
    return Path.home() / ".config" / "hegemon-agent" / "settings.json"


def load_settings():
    try:
        data = json.loads(settings_file().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data):
    f = settings_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(str(tmp), str(f))


class Config(object):
    def __init__(self):
        self.user = _user()
        self.lab_script = Path(os.environ.get("HEGEMON_BOOLEANLAB_SCRIPT") or _first(LAB_SCRIPT_CANDIDATES))
        self.genome_dir = Path(os.environ.get("HEGEMON_GENOME_DIR") or _first(GENOME_CANDIDATES))

        runs = os.environ.get("HEGEMON_RUNS")
        if runs:
            self.runs_root = Path(runs).expanduser()
        else:
            self.runs_root = Path.home() / "hegemon-agent-runs"
            for base in ("/booleanfs2/sahoo/Data/BooleanLab", "/mnt/booleanfs2/sahoo/Data/BooleanLab"):
                if (Path(base) / self.user).is_dir():
                    self.runs_root = Path(base) / self.user / "agent_runs"
                    break

        self.site_root = Path(os.environ.get("HEGEMON_SITE_ROOT") or Path.home() / "public_html" / "Hegemon").expanduser()
        self.site_url = (os.environ.get("HEGEMON_SITE_URL")
                         or "http://hegemon.ucsd.edu/~{}/Hegemon".format(self.user)).rstrip("/")

        self.key_file = Path.home() / ".config" / "hegemon-agent" / "openai.key"  # same file V0-V6 used
        self.model = os.environ.get("HEGEMON_MODEL", "gpt-5.6-sol")
        self.fast_model = os.environ.get("HEGEMON_FAST_MODEL", "gpt-5.6-luna")
        settings = load_settings()
        mode = (os.environ.get("HEGEMON_MODE") or settings.get("mode") or "").strip().lower()
        self.mode = mode if mode in MODES else None          # None: not chosen yet (cloud until it is)
        self.local_url = (os.environ.get("HEGEMON_LOCAL_URL") or settings.get("local_url")
                          or "http://localhost:11434/v1").rstrip("/")
        self.local_model = os.environ.get("HEGEMON_LOCAL_MODEL") or settings.get("local_model") or "qwen2.5-coder-16k"
        self.cloud_model, self.cloud_fast_model = self.model, self.fast_model
        self.apply_mode()
        self.reasoning = (os.environ.get("HEGEMON_REASONING") or "").strip() or None  # None: medium, then high
        self.ai_timeout = _num("HEGEMON_AI_TIMEOUT", 900)
        self.attempts = int(_num("HEGEMON_ATTEMPTS", 5))

        self.max_file_gb = _num("HEGEMON_MAX_FILE_GB", 5)
        self.max_total_gb = _num("HEGEMON_MAX_TOTAL_GB", 20)
        self.script_timeout = _num("HEGEMON_SCRIPT_TIMEOUT", 4 * 3600)
        self.lab_timeout = _num("HEGEMON_LAB_TIMEOUT", 12 * 3600)

    def apply_mode(self):
        """local: the local model for everything. mixed: local for ranking and file picking (the fast-model
        calls), OpenAI for the conversions. cloud (or not chosen): OpenAI, exactly as before."""
        self.model, self.fast_model = self.cloud_model, self.cloud_fast_model
        if self.mode == "local":
            self.model = self.fast_model = self.local_model
        elif self.mode == "mixed":
            self.fast_model = self.local_model

    def is_local(self, model):
        return self.mode in ("local", "mixed") and model == self.local_model

    def needs_key(self):
        return self.mode != "local"
