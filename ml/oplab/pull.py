"""MacBook: fetch the Mac mini's ml/shipped/ and install it where the app build picks it up:
data/cards/{printings.f32, printings.meta.json, catalog.json} (copied into the bundle by the build
phase) and apps/ios/OnePieceAR/Resources/Models/CardEmbedder.mlpackage (removed for a feature-print
shipment, or the app would use the stale model and ignore the new index). Nothing is installed unless
the fetched shipment validates.

Where ml/shipped/ comes from (ml/remote.env):
- MINI_SHIPPED=/Volumes/op-tcg-ar/ml/shipped  the mini's repo shared over File Sharing and mounted in
  Finder (no SSH needed; preferred)
- MINI_HOST + MINI_REPO                       rsync over SSH (MINI_HOST=local: this Mac's own ml/shipped)"""

import os
import shutil
import subprocess
from pathlib import Path

from . import paths, remote, shipped


class PullError(Exception):
    pass


def fetch(host: str, repo: str, staging: Path, run=subprocess.run, local_shipped: Path = paths.SHIPPED) -> None:
    shutil.rmtree(staging, ignore_errors=True)
    if host == "local":
        if not local_shipped.exists():
            raise PullError(f"nothing shipped in {local_shipped}; run make ship-baseline (or make ship) first")
        shutil.copytree(local_shipped, staging)
        return
    staging.mkdir(parents=True)
    try:
        run(["rsync", "-a", "--delete", f"{host}:{repo.rstrip('/')}/ml/shipped/", f"{staging}/"], check=True)
    except subprocess.CalledProcessError as error:
        hints = {23: "nothing shipped on the mini? run make ship-baseline there",
                 255: "ssh failed: is Remote Login on and the key authorized?"}
        hint = hints.get(error.returncode, "is Remote Login on and the key authorized?")
        raise PullError(f"rsync from {host} failed ({error.returncode}); {hint}") from error
    except OSError as error:
        raise PullError(f"rsync not found or not runnable ({error})") from error


def source(env: dict[str, str]) -> tuple:
    """("mount", path) when MINI_SHIPPED is set, else ("ssh", host, repo)."""
    if env.get("MINI_SHIPPED"):
        return ("mount", Path(env["MINI_SHIPPED"]).expanduser())
    host, repo = env.get("MINI_HOST", ""), env.get("MINI_REPO", "")
    if not host or not repo:
        raise remote.RemoteConfigError(
            f"set MINI_SHIPPED (the mini's repo mounted over File Sharing) or MINI_HOST and MINI_REPO (SSH) "
            f"in {paths.REMOTE_ENV} (copy ml/remote.env.example)")
    return ("ssh", host, repo)


def fetch_mounted(shipped_dir: Path, staging: Path) -> None:
    """Copies ml/shipped/ from the mini's repo, mounted over File Sharing (SMB)."""
    if not shipped_dir.parent.exists():
        raise PullError(f"{shipped_dir.parent} isn't there: mount the mini's repo first "
                        "(Finder → Go → Connect to Server → smb://<mini>.local → op-tcg-ar)")
    fetch("local", "", staging, local_shipped=shipped_dir)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def install(staged: Path, data_cards: Path = paths.DATA_CARDS, app_models: Path = paths.APP_MODELS) -> list[str]:
    """Validate, prepare every new file under a temporary name, and only then swap them in (model first),
    so an interruption while preparing leaves the existing index and model untouched."""
    problems = shipped.validate(staged)
    if problems:
        raise PullError("not installed: " + "; ".join(problems))
    target = app_models / shipped.MODEL_DIR
    incoming_model = app_models / (shipped.MODEL_DIR + ".incoming")
    old_model = app_models / (shipped.MODEL_DIR + ".old")
    incoming = {name: data_cards / (name + ".incoming") for name in shipped.FILES}
    has_model = (staged / shipped.MODEL_DIR).exists()
    try:
        for path in (incoming_model, old_model, *incoming.values()):
            _remove(path)
        for name, temp in incoming.items():
            shutil.copy2(staged / name, temp)
        if has_model:
            app_models.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staged / shipped.MODEL_DIR, incoming_model)
    except (OSError, shutil.Error) as error:
        for path in (incoming_model, *incoming.values()):
            _remove(path)
        raise PullError(f"not installed, existing files untouched: {error}") from error

    actions = []
    had_model = target.exists()
    if had_model:
        target.rename(old_model)
    if has_model:
        incoming_model.rename(target)
        actions.append(f"{shipped.MODEL_DIR} -> {app_models}")
    elif had_model:
        actions.append(f"removed old {shipped.MODEL_DIR}")
    if had_model:
        actions.append("Xcode may keep the old compiled model: Product → Clean Build Folder (⇧⌘K) before rebuilding")
    _remove(old_model)
    for name, temp in incoming.items():
        os.replace(temp, data_cards / name)
        actions.append(f"{name} -> {data_cards}")
    return actions


def main() -> None:
    where = source(remote.read_env())
    staging = paths.ML / ".pull-staging"
    try:
        if where[0] == "mount":
            fetch_mounted(where[1], staging)
        else:
            fetch(where[1], where[2], staging)
        actions = install(staging)
        info = shipped.read(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for action in actions:
        print(f"  {action}")
    print(f"installed {info['name']} ({info['backend']}, shipped {info['shipped']}); rebuild the app in Xcode")
    print("note: data/cards/catalog.json may show as modified in git if the mini's catalog is newer")
