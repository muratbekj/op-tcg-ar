"""Mac mini setup check (`make mini-doctor`): prints ✓ / ✗ / ? per item with how to fix it.
The checks run against a probe so they can be tested without a second Mac."""

import importlib.util
import os
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
        return os.path.realpath(paths.INBOX) in {os.path.realpath(p) for p in shared}

    def ssh_listening(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 22), timeout=0.5):
                return True
        except OSError:
            return False

    def smb_listening(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 445), timeout=0.5):
                return True
        except OSError:
            return False

    def login_has(self, cmd: str) -> bool | None:
        """Whether `cmd` is on PATH in a login shell (train-remote runs under `zsh -lc`)."""
        try:
            return subprocess.run(["zsh", "-lc", f"command -v {cmd}"], capture_output=True).returncode == 0
        except OSError:
            return None

    def authorized_keys(self) -> bool | None:
        keys = Path.home() / ".ssh" / "authorized_keys"
        try:
            return keys.exists() and keys.read_text().strip() != ""
        except (OSError, UnicodeError):
            return None

    def index_rows(self) -> int:
        try:
            return len(io.read_json(paths.INDEX_META)["rows"]) if paths.INDEX_META.exists() else 0
        except (OSError, ValueError, KeyError, TypeError):
            return 0


def _login_detail(found: bool | None, on_path: bool, install: str) -> str:
    if found is None:
        return "couldn't run a login shell (zsh -lc)"
    if found:
        return "found"
    return "installed but not on the login PATH: add it to ~/.zprofile" if on_path else install


def checks(probe) -> list[Check]:
    xcode = probe.xcode_version()
    shared = probe.inbox_shared()
    rows = probe.index_rows()
    inbox_exists = probe.inbox_exists()
    ssh = probe.ssh_listening()
    keys = probe.authorized_keys()
    smb = probe.smb_listening() if shared else None
    has_uv, has_tmux = probe.login_has("uv"), probe.login_has("tmux")
    training = probe.has_module("torch") and probe.has_module("coremltools")
    return [
        Check("macOS", True, f"macOS {probe.macos_version()}: keep the MacBook on the same major version"),
        Check("Xcode", xcode is not None, xcode or "install Xcode (same version as the MacBook): evals run the Swift cardvision CLI"),
        Check("uv", has_uv, _login_detail(has_uv, probe.which("uv"), "install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")),
        Check("tmux", has_tmux, _login_detail(has_tmux, probe.which("tmux"),
                                              "brew install tmux (train-remote runs training inside it)")),
        Check("training extras", training, "torch + coremltools" if training else "make ml-setup-train"),
        Check("inbox folder", inbox_exists, str(paths.INBOX) if inbox_exists else "mkdir ~/oplab-inbox"),
        Check("inbox shared (SMB)", None if shared is None else bool(shared and smb),
              "couldn't read `sharing -l`; check System Settings → General → Sharing → File Sharing" if shared is None
              else ("shared" if smb else "turn on File Sharing (System Settings → General → Sharing)") if shared
              else "System Settings → General → Sharing → File Sharing → + → ~/oplab-inbox"),
        Check("Remote Login (SSH)", ssh,
              "on" if ssh else "System Settings → General → Sharing → Remote Login"),
        Check("MacBook key", keys,
              {True: "authorized", None: "couldn't read ~/.ssh/authorized_keys"}
              .get(keys, "on the MacBook: ssh-copy-id <user>@<this-mac>.local")),
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
