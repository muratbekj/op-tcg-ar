"""Export a trained embedder to Core ML as CardEmbedder.mlpackage.

The model takes an RGB image (Vision scales the canonical card to fit the input) and outputs an
L2-normalized `embedding`. Its version string ends up in the backend ID ("coreml:CardEmbedder@<version>"),
so an index built with one model is never matched against another.

After exporting: regenerate references with the same model
(generate_embeddings.py --model ml/models/<name>/CardEmbedder.mlpackage), then evaluate.
"""

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import torch

from . import paths
from .train import Embedder

MODEL_NAME = "CardEmbedder"


def model_version(name: str) -> str:
    """The Core ML model's version string; the app's backend ID becomes `coreml:CardEmbedder@<version>`,
    which `ml/shipped/shipped.json` and index metadata must match."""
    return name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="card-embedder", help="training run under ml/models/")
    parser.add_argument("--install", action="store_true",
                        help="copy into the app's Resources/Models (gitignored) so Xcode bundles it")
    args = parser.parse_args(argv)

    run_dir = paths.MODELS / args.name
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    model = Embedder(checkpoint["dim"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    width, height = checkpoint["input_size"]
    example = torch.rand(1, 3, height, width)
    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    version = model_version(args.name)
    mlmodel.version = version
    mlmodel.short_description = f"Card embedding ({checkpoint['dim']}-d) trained on {len(checkpoint['classes'])} printings"

    out = run_dir / f"{MODEL_NAME}.mlpackage"
    if out.exists():
        shutil.rmtree(out)
    mlmodel.save(str(out))
    print(f"saved {out} (version {version})")

    if args.install:
        target = paths.APP_MODELS / out.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(out, target)
        print(f"installed {target}; regenerate data/cards/printings.f32 with --model before building the app")


if __name__ == "__main__":
    main()
