"""Model versions under ml/models/<name>/ (v0 = Apple's Vision feature print, v1… = fine-tuned).

`eval_version` builds the full-catalog index with that version's embedder, scores it on the latest
frozen real test set (or the synthetic/photo manifest with diagnostic=True, or the unfrozen test pool
with scans="test", a preliminary real-scan check), and writes metrics.json
and MODEL_CARD.md. `ship_version` only ships versions evaluated on the current frozen set.
"""

import argparse
import platform
import shutil
from pathlib import Path

from . import dataset, doctor, embeddings, evaluate, io, paths, remote, shipped, testsets

FEATURE_PRINT_VERSION = "v0"
MODEL_DIR = "CardEmbedder.mlpackage"


class RegistryError(Exception):
    pass


def version_dir(name: str) -> Path:
    if not remote.SAFE_NAME.fullmatch(name) or ".." in name:
        raise ValueError(f"version name must be letters, digits, '.', '_' or '-' (got {name!r})")
    return paths.MODELS / name


def environment() -> dict:
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {"macos": platform.mac_ver()[0] or "unknown", "swift": doctor.SystemProbe().swift_version() or "unknown",
            "torch": torch_version}


def _pct(value) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def _with_ci(value, ci) -> str:
    if value is None or not ci or ci[0] is None:
        return _pct(value)
    return f"{_pct(value)} ({ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"


def render_card(name: str, training: dict | None, metrics: dict, environment: dict) -> str:
    s = metrics["summary"]
    lines = [f"# Model card: {name}", "",
             f"- Backend: `{metrics['backend']}`"]
    if training is None:
        lines += ["- Model: Apple Vision feature print (revision 2), pretrained; no training (the baseline)."]
    else:
        c = training["counts"]
        lines += [f"- Parent: {training['parent']} · trained {training['created']} · {training['epochs']} epochs",
                  f"- Training data: {c['printings']} catalog printings (augmented art) + {c['scans']} real scans of "
                  f"{c['scan_printings']} printings (×{c['scan_weight']} oversampled; {c['scans_skipped']} scans without "
                  "art skipped), group-aware batches",
                  f"- Synthetic validation top-1 at the end of training: {_pct(training.get('synthetic_val_top1'))}"]
    if metrics["testset"]:
        lines += [f"- Test set: `{metrics['testset']}` (frozen real scans; printings never used for training)"]
    elif metrics.get("source") == testsets.POOL_NAME:
        lines += [f"- Test set: none frozen. **Preliminary** run on the unfrozen test pool ({s.get('n', '?')} real scans of "
                  "printings never used for training): not shippable, and not comparable over time as the pool grows."]
    else:
        lines += ["- Test set: none. **Diagnostic** run on the synthetic/photo manifest: not shippable, "
                  "not a headline number."]
    lines += ["", "| Metric | Value (95% CI) |", "| --- | --- |",
              f"| Top-1 printing | {_with_ci(s.get('top1'), s.get('top1_ci'))} |",
              f"| OCR accuracy | {_with_ci(s.get('ocr_accuracy'), s.get('ocr_accuracy_ci'))} |",
              f"| Within-group top-1 | {_with_ci(s.get('within_group'), s.get('within_group_ci'))} |",
              f"| Detection | {_with_ci(s.get('detection'), s.get('detection_ci'))} |", ""]
    by_method = metrics.get("groups", {}).get("method", {})
    if by_method:
        lines += ["| Method | n | Top-1 |", "| --- | --- | --- |"]
        lines += [f"| {method} | {v['n']} | {_pct(v['recall'])} |" for method, v in sorted(by_method.items())]
        lines.append("")
    lines += ["## Environment", "",
              f"- macOS {environment['macos']} · {environment['swift']}"
              + (f" · torch {environment['torch']}" if environment.get("torch") else ""), "",
              "## Caveats", "",
              "- Real scans come from one collection and one iPhone; other cards, sleeves, and lighting may differ.",
              "- Train/test split is by printing: test printings were never photographed for training, so the score "
              "measures unseen cards, but test cards share sets, layouts, and photo conditions with training ones.",
              "- The eval runs Vision/Core ML on macOS; the phone runs iOS, whose results can differ slightly.", ""]
    return "\n".join(lines)


def eval_version(name: str, diagnostic: bool = False, scans: str | None = None) -> dict:
    directory = version_dir(name)
    model = directory / MODEL_DIR
    if name != FEATURE_PRINT_VERSION and not model.exists():
        raise RegistryError(f"no {model}; train and export first: make train NAME={name} && make export NAME={name}")
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "printings.f32"

    embed_args = ["--out", str(index)]
    if name != FEATURE_PRINT_VERSION:
        embed_args += ["--model", str(model)]
    embeddings.main(embed_args)

    eval_args = ["--name", name, "--index", str(index)]
    if name != FEATURE_PRINT_VERSION:
        eval_args += ["--model", str(model)]
    testset = None
    if scans:
        eval_args += ["--scans", scans]
    elif not diagnostic:
        testset = testsets.load("latest")["name"]
        eval_args += ["--testset", testset]
    run_dir = evaluate.main(eval_args)

    run_metrics = io.read_json(Path(run_dir) / "metrics.json")
    source = "frozen" if testset else testsets.POOL_NAME if scans else "diagnostic"
    stored = {"name": name, "testset": testset, "source": source, "run": str(run_dir),
              "backend": io.read_json(directory / "printings.meta.json")["backend"],
              "summary": run_metrics["summary"], "groups": run_metrics.get("groups", {})}
    io.write_json(directory / "metrics.json", stored)
    training = io.read_json(directory / "training.json") if (directory / "training.json").exists() else None
    (directory / "MODEL_CARD.md").write_text(render_card(name, training, stored, environment()))
    return stored


def ship_version(name: str, docs_models: Path = paths.REPO / "docs" / "models") -> dict:
    """Stage version `name` into ml/shipped/ for the app, only if it was evaluated on the current
    frozen test set. Copies its model card to docs/models/<name>.md (commit it)."""
    directory = version_dir(name)
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        raise RegistryError(f"{name} hasn't been evaluated: make eval NAME={name}")
    try:
        current = testsets.load("latest")["name"]
    except FileNotFoundError as error:
        raise RegistryError(f"no frozen test set yet ({error}); ship-baseline ships v0 for app testing") from error
    evaluated = io.read_json(metrics_path).get("testset")
    if evaluated != current:
        where = evaluated or ("the unfrozen test pool" if io.read_json(metrics_path).get("source") == testsets.POOL_NAME
                              else "the diagnostic manifest")
        raise RegistryError(f"{name} was evaluated on {where}, not the current "
                            f"frozen set {current}: make eval NAME={name}")
    model = directory / MODEL_DIR
    has_model = model.exists()
    info = shipped.stage(name, directory / "printings.f32", directory / "printings.meta.json", paths.FULL_CATALOG,
                         labels=len(dataset.scan_records()), model=model if has_model else None,
                         model_version=name if has_model else None)
    docs_models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directory / "MODEL_CARD.md", docs_models / f"{name}.md")
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="evaluate a version on the latest frozen test set and write its model card")
    ev.add_argument("name")
    ev.add_argument("--diagnostic", action="store_true", help="synthetic/photo manifest instead (not shippable)")
    ev.add_argument("--scans", choices=["test"], help="the unfrozen test pool instead (preliminary; not shippable)")
    sh = sub.add_parser("ship", help="ship an evaluated version to ml/shipped/ (the app's next pull-model)")
    sh.add_argument("name")
    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            stored = eval_version(args.name, args.diagnostic, args.scans)
            where = stored["testset"] or {"test-pool": "the unfrozen test pool (preliminary)"}.get(
                stored["source"], "the diagnostic manifest")
            print(f"{args.name}: top-1 {_with_ci(stored['summary']['top1'], stored['summary'].get('top1_ci'))} "
                  f"on {where}; card: {version_dir(args.name) / 'MODEL_CARD.md'}")
        elif args.command == "ship":
            info = ship_version(args.name)
            print(f"shipped {info['name']} -> ml/shipped/; card -> docs/models/{args.name}.md (commit it); "
                  "then make pull-model and rebuild the app")
    except (ValueError, RegistryError, FileNotFoundError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
