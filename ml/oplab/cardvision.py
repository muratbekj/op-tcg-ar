"""Wrapper around the Swift `cardvision` CLI (apps/ios/Packages/OnePieceKit), which runs the exact
recognition code the phone runs. Python never re-implements embedding or matching."""

import json
import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path

from . import io, paths

XCODE = "/Applications/Xcode.app/Contents/Developer"


def _env() -> dict:
    env = dict(os.environ)
    # The CLI needs the full Xcode toolchain even when xcode-select points at the Command Line Tools.
    if "DEVELOPER_DIR" not in env and Path(XCODE).exists():
        env["DEVELOPER_DIR"] = XCODE
    return env


@cache
def binary() -> Path:
    """Builds the CLI in release mode (incremental) and returns its path."""
    subprocess.run(
        ["swift", "build", "-c", "release", "--product", "cardvision", "--package-path", str(paths.SWIFT_PACKAGE)],
        check=True, env=_env(), stdout=subprocess.DEVNULL)
    out = subprocess.run(
        ["swift", "build", "-c", "release", "--show-bin-path", "--package-path", str(paths.SWIFT_PACKAGE)],
        check=True, env=_env(), capture_output=True, text=True).stdout.strip()
    return Path(out) / "cardvision"


def embed(entries: list[dict], out: Path, model: Path | None = None, min_similarity: float | None = None) -> dict:
    """entries: [{"printingId", "path"}] with absolute paths. Writes `out` and `<out stem>.meta.json`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "refs.json"
        io.write_json(manifest, entries)
        command = [str(binary()), "embed", "--manifest", str(manifest), "--out", str(out),
                   "--meta", str(meta_path(out))]
        if model:
            command += ["--model", str(model)]
        if min_similarity is not None:
            command += ["--min-similarity", str(min_similarity)]
        result = subprocess.run(command, check=True, env=_env(), stdout=subprocess.PIPE, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def match(index: Path, queries: list[dict], mode: str, k: int = 5, ocr: bool = True,
          model: Path | None = None) -> list[dict]:
    """queries: [{"id", "path"}]. mode "photo" detects the card first; "card" treats the image as the card."""
    if not queries:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        manifest, out = Path(tmp) / "queries.json", Path(tmp) / "predictions.jsonl"
        io.write_json(manifest, queries)
        command = [str(binary()), "match", "--index", str(index), "--meta", str(meta_path(index)),
                   "--queries", str(manifest), "--out", str(out), "--mode", mode, "--k", str(k)]
        if not ocr:
            command.append("--no-ocr")
        if model:
            command += ["--model", str(model)]
        subprocess.run(command, check=True, env=_env())
        return io.read_jsonl(out)


def meta_path(index: Path) -> Path:
    return index.with_suffix(".meta.json")
