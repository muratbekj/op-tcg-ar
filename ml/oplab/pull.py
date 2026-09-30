"""MacBook: fetch the Mac mini's ml/shipped/ and install it where the app build picks it up:
data/cards/{printings.f32, printings.meta.json, catalog.json} (copied into the bundle by the build
phase) and apps/ios/OnePieceAR/Resources/Models/CardEmbedder.mlpackage (removed for a feature-print
shipment, or the app would use the stale model and ignore the new index). Nothing is installed unless
the fetched shipment validates."""

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
        raise PullError(f"rsync from {host} failed ({error.returncode}); is Remote Login on and the key authorized?")


def install(staged: Path, data_cards: Path = paths.DATA_CARDS, app_models: Path = paths.APP_MODELS) -> list[str]:
    problems = shipped.validate(staged)
    if problems:
        raise PullError("not installed: " + "; ".join(problems))
    actions = []
    for name in shipped.FILES:
        shutil.copy2(staged / name, data_cards / name)
        actions.append(f"{name} -> {data_cards}")
    target = app_models / shipped.MODEL_DIR
    if target.exists():
        shutil.rmtree(target)
        actions.append(f"removed old {shipped.MODEL_DIR}")
    if (staged / shipped.MODEL_DIR).exists():
        app_models.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged / shipped.MODEL_DIR, target)
        actions.append(f"{shipped.MODEL_DIR} -> {app_models}")
    return actions


def main() -> None:
    host, repo = remote.mini()
    staging = paths.ML / ".pull-staging"
    fetch(host, repo, staging)
    actions = install(staging)
    info = shipped.read(staging)
    for action in actions:
        print(f"  {action}")
    print(f"installed {info['name']} ({info['backend']}, shipped {info['shipped']}); rebuild the app in Xcode")
