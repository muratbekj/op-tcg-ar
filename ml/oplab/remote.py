"""Working across the two Macs, from the MacBook: ml/remote.env names the Mac mini.

ml/remote.env (gitignored; copy ml/remote.env.example):
    MINI_HOST=murat@mac-mini.local     # ssh target; "local" = this Mac is also the ML Mac
    MINI_REPO=~/github/op-tcg-ar       # the repo's path on the mini
"""

import argparse
import re
import shlex
from pathlib import Path

from . import paths


class RemoteConfigError(Exception):
    pass


def read_env(path: Path = paths.REMOTE_ENV) -> dict[str, str]:
    """KEY=VALUE lines; blank lines, # comments and lines without '=' are ignored; quotes stripped."""
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mini(path: Path = paths.REMOTE_ENV) -> tuple[str, str]:
    env = read_env(path)
    host, repo = env.get("MINI_HOST", ""), env.get("MINI_REPO", "")
    if not host or not repo:
        raise RemoteConfigError(f"set MINI_HOST and MINI_REPO in {path} (copy ml/remote.env.example)")
    return host, repo


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def train_command(host: str, repo: str, name: str, extra: str = "") -> list[str]:
    """ssh command that starts `make train NAME=<name>` on the mini inside a detached tmux session,
    logging to ml/runs/train-<name>.log. A login shell (zsh -lc) so Homebrew's uv/tmux are on PATH."""
    if not SAFE_NAME.fullmatch(name) or ".." in name:
        raise ValueError(f"NAME must be letters, digits, '.', '_' or '-' (got {name!r})")
    args = f" ARGS={shlex.quote(extra)}" if extra else ""
    job = f"cd {repo} && mkdir -p ml/runs && make train NAME={name}{args} 2>&1 | tee ml/runs/train-{name}.log"
    tmux = f"tmux new-session -d -s train-{name} {shlex.quote(job)}"
    return ["ssh", host, f"zsh -lc {shlex.quote(tmux)}"]


def main(argv: list[str] | None = None) -> None:
    from . import pull

    parser = argparse.ArgumentParser(description="Two-Mac commands (see ml/README.md, 'Two Macs').")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pull-model", help="MacBook: fetch ml/shipped/ from the mini and install it for the app build")
    sub.add_parser("doctor", help="Mac mini: check the one-time setup")
    train = sub.add_parser("train-remote", help="MacBook: start make train NAME=… on the mini inside tmux")
    train.add_argument("name")
    train.add_argument("--args", default="", help="extra train_embedding.py arguments")
    args = parser.parse_args(argv)

    try:
        if args.command == "pull-model":
            pull.main()
        elif args.command == "doctor":
            from . import doctor

            text, code = doctor.render(doctor.checks(doctor.SystemProbe()))
            print(text)
            raise SystemExit(code)
        elif args.command == "train-remote":
            import subprocess

            host, repo = mini()
            try:
                command = train_command(host, repo, args.name, args.args)
            except ValueError as error:
                raise SystemExit(str(error))
            subprocess.run(command, check=True)
            print(f"training {args.name} started on {host} in tmux session train-{args.name}")
            print(f"  watch:  ssh -t {host} tmux attach -t train-{args.name}   (detach: Ctrl-b d)")
            print(f"  log:    {repo}/ml/runs/train-{args.name}.log")
    except (RemoteConfigError, pull.PullError) as error:
        raise SystemExit(str(error))
