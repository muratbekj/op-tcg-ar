"""Mac mini setup check (`make mini-doctor`): prints ✓ / ✗ / ? per item with how to fix it.
The checks run against a probe so they can be tested without a second Mac."""

import importlib.util
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import io, paths

FULL_CATALOG_ROWS = 4000


@dataclass
class Check:
    name: str
    ok: bool | None  # None: couldn't tell
    detail: str


class SystemProbe:
    def macos_version(self) -> str:
        return platform.mac_ver()[0] or "unknown"

    def xcode_version(self) -> str | None:
        try:
            out = subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.splitlines()[0] if out else None

    def which(self, cmd: str) -> bool:
        return shutil.which(cmd) is not None

    def has_module(self, name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def inbox_exists(self) -> bool:
        return paths.INBOX.is_dir()

    def inbox_shared(self) -> bool | None:
        try:
            out = subprocess.run(["sharing", "-l"], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        shared = {line.split(":", 1)[1].strip() for line in out.splitlines() if line.strip().startswith("path:")}
        return str(paths.INBOX) in shared

    def ssh_listening(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 22), timeout=0.5):
                return True
        except OSError:
            return False

    def authorized_keys(self) -> bool:
        keys = Path.home() / ".ssh" / "authorized_keys"
        return keys.exists() and keys.read_text().strip() != ""

    def index_rows(self) -> int:
        return len(io.read_json(paths.INDEX_META)["rows"]) if paths.INDEX_META.exists() else 0


def checks(probe) -> list[Check]:
    xcode = probe.xcode_version()
    shared = probe.inbox_shared()
    rows = probe.index_rows()
    training = probe.has_module("torch") and probe.has_module("coremltools")
    return [
        Check("macOS", True, f"macOS {probe.macos_version()}: keep the MacBook on the same major version"),
        Check("Xcode", xcode is not None, xcode or "install Xcode (same version as the MacBook): evals run the Swift cardvision CLI"),
        Check("uv", probe.which("uv"), "found" if probe.which("uv") else "install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"),
        Check("tmux", probe.which("tmux"), "found" if probe.which("tmux") else "brew install tmux (train-remote runs training inside it)"),
        Check("training extras", training, "torch + coremltools" if training else "make ml-setup-train"),
        Check("inbox folder", probe.inbox_exists(), str(paths.INBOX) if probe.inbox_exists() else "mkdir ~/oplab-inbox"),
        Check("inbox shared (SMB)", shared,
              {True: "shared", None: "couldn't read `sharing -l`; check System Settings → General → Sharing → File Sharing"}
              .get(shared, "System Settings → General → Sharing → File Sharing → + → ~/oplab-inbox")),
        Check("Remote Login (SSH)", probe.ssh_listening(),
              "on" if probe.ssh_listening() else "System Settings → General → Sharing → Remote Login"),
        Check("MacBook key", probe.authorized_keys(),
              "authorized" if probe.authorized_keys() else "on the MacBook: ssh-copy-id <user>@<this-mac>.local"),
        Check("full-catalog index", rows >= FULL_CATALOG_ROWS,
              f"{rows} rows" if rows >= FULL_CATALOG_ROWS else
              f"{rows} rows; run fetch_cards.py --art all, then generate_embeddings.py --min-similarity 0.8"),
    ]


def render(results: list[Check]) -> tuple[str, int]:
    marks = {True: "✓", False: "✗", None: "?"}
    lines = [f"{marks[c.ok]} {c.name}: {c.detail}" for c in results]
    failed = sum(c.ok is False for c in results)
    lines.append("all set" if not failed else f"{failed} to fix")
    return "\n".join(lines), 1 if failed else 0
