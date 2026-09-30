"""Working across the two Macs, from the MacBook: ml/remote.env names the Mac mini.

ml/remote.env (gitignored; copy ml/remote.env.example):
    MINI_HOST=murat@mac-mini.local     # ssh target; "local" = this Mac is also the ML Mac
    MINI_REPO=~/github/op-tcg-ar       # the repo's path on the mini
"""

import argparse
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


def main(argv: list[str] | None = None) -> None:
    from . import pull

    parser = argparse.ArgumentParser(description="Two-Mac commands (see ml/README.md, 'Two Macs').")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pull-model", help="MacBook: fetch ml/shipped/ from the mini and install it for the app build")
    sub.add_parser("doctor", help="Mac mini: check the one-time setup")
    args = parser.parse_args(argv)

    try:
        if args.command == "pull-model":
            pull.main()
        elif args.command == "doctor":
            from . import doctor

            text, code = doctor.render(doctor.checks(doctor.SystemProbe()))
            print(text)
            raise SystemExit(code)
    except (RemoteConfigError, pull.PullError) as error:
        raise SystemExit(str(error))
