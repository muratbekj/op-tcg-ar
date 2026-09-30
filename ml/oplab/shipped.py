"""ml/shipped/: the one version the app should bundle, rsynced from the Mac mini to the MacBook.

Contents: printings.f32 + printings.meta.json (the reference index), catalog.json, shipped.json
({name, backend, modelVersion, shipped, labels}), and CardEmbedder.mlpackage when the backend is a
Core ML model. A feature-print shipment has no model. `validate` enforces that the index, the model,
and shipped.json agree, because the app silently ignores an index built by a different embedder.
"""

import argparse
import shutil
from datetime import date
from pathlib import Path

from . import io, paths

FEATURE_PRINT = "vision-featureprint-r2"
MODEL_DIR = "CardEmbedder.mlpackage"
FILES = ("printings.f32", "printings.meta.json", "catalog.json")
INFO = "shipped.json"


def read(directory: Path = paths.SHIPPED) -> dict | None:
    path = directory / INFO
    return io.read_json(path) if path.exists() else None


def validate(directory: Path) -> list[str]:
    info = read(directory)
    if info is None:
        return [f"no {INFO} in {directory}"]
    problems = [f"missing {name}" for name in FILES if not (directory / name).exists()]
    meta_path = directory / "printings.meta.json"
    if meta_path.exists() and io.read_json(meta_path)["backend"] != info["backend"]:
        problems.append(f"index backend {io.read_json(meta_path)['backend']} doesn't match shipped backend {info['backend']}")
    if (directory / MODEL_DIR).exists():
        expected = f"coreml:CardEmbedder@{info.get('modelVersion')}"
        if info["backend"] != expected:
            problems.append(f"model version {info.get('modelVersion')} doesn't match backend {info['backend']}")
    elif info["backend"] != FEATURE_PRINT:
        problems.append(f"backend {info['backend']} needs {MODEL_DIR}, which isn't shipped")
    return problems


def stage(name: str, index: Path, meta: Path, catalog: Path, labels: int, directory: Path = paths.SHIPPED,
          model: Path | None = None, model_version: str | None = None, today: str | None = None) -> dict:
    """Replaces `directory` with this version, atomically: builds it next to the old one, validates,
    then swaps. A shipment that wouldn't validate raises ValueError and the old one stays."""
    staging = directory.with_name(directory.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copy2(index, staging / "printings.f32")
    shutil.copy2(meta, staging / "printings.meta.json")
    shutil.copy2(catalog, staging / "catalog.json")
    if model is not None:
        shutil.copytree(model, staging / MODEL_DIR)
    info = {"name": name, "backend": io.read_json(meta)["backend"], "modelVersion": model_version,
            "shipped": today or date.today().isoformat(), "labels": labels}
    io.write_json(staging / INFO, info)
    problems = validate(staging)
    if problems:
        shutil.rmtree(staging)
        raise ValueError("; ".join(problems))
    shutil.rmtree(directory, ignore_errors=True)
    staging.rename(directory)
    return info


def status_line(labeled_now: int, directory: Path = paths.SHIPPED) -> str:
    info = read(directory)
    if info is None:
        return "shipped: none yet (make ship-baseline records v0)"
    return (f"shipped: {info['name']} ({info['backend']}, {info['shipped']}), "
            f"{labeled_now - info['labels']} new labels since")


def main(argv: list[str] | None = None) -> None:
    from . import dataset

    parser = argparse.ArgumentParser(description="Stage what the app should bundle into ml/shipped/.")
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline", help="ship the current Vision feature-print index (the v0 baseline)")
    baseline.add_argument("--name", default="v0")
    args = parser.parse_args(argv)

    if args.command == "baseline":
        if not paths.INDEX_META.exists():
            raise SystemExit("no index; run generate_embeddings.py --min-similarity 0.8 first")
        if io.read_json(paths.INDEX_META)["backend"] != FEATURE_PRINT:
            raise SystemExit(f"data/cards index isn't the feature print; rebuild it without --model to ship {args.name}")
        try:
            info = stage(args.name, paths.INDEX, paths.INDEX_META, paths.FULL_CATALOG, labels=len(dataset.scan_records()))
        except ValueError as error:
            raise SystemExit(f"not shipped: {error}")
        print(f"shipped {info['name']} ({info['backend']}) -> {paths.SHIPPED.relative_to(paths.REPO)}/; "
              f"on the MacBook: make pull-model")
