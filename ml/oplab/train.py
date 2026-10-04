"""Fine-tune a card embedding model (only worth it once Vision feature prints plateau).

Every printing is a class. Items are augmented views of the catalog art plus the user's labeled
train-split scans, oversampled (`trainset.build_items`); views come from `synth.augment_card`
(residual perspective, lighting, glare, blur, watermark band removal), and a CosFace head pulls views
of the same printing together. Batches are group-aware, so same-code printings are separated against
each other. The embedder normalizes its own input, so the exported Core ML model takes plain RGB in
[0, 1]. `training.json` next to the checkpoint records what the run was made of.
"""

import argparse
import random
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from . import dataset, io, paths, remote, shipped, synth, trainset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Embedder(nn.Module):
    def __init__(self, dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(960, dim)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = self.pool(self.features(x)).flatten(1)
        return F.normalize(self.head(x), dim=1)


class CosFace(nn.Module):
    def __init__(self, dim: int, classes: int, scale: float = 30.0, margin: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(classes, dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = embeddings @ F.normalize(self.weight, dim=1).T
        margin = F.one_hot(labels, cosine.shape[1]) * self.margin
        return F.cross_entropy(self.scale * (cosine - margin), labels)


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, np.float32) / 255).permute(2, 0, 1)


class TrainingViews(Dataset):
    """One augmented view per item (art or real scan); images are downscaled once and cached."""

    def __init__(self, items: list[trainset.Item]):
        self.items = items
        self.cache: dict[str, Image.Image] = {}

    def image(self, path: str) -> Image.Image:
        # Downscale once: augmentation works at roughly the canonical card size.
        if path not in self.cache:
            self.cache[path] = Image.open(path).convert("RGB").resize((315, 440), Image.Resampling.BILINEAR)
        return self.cache[path]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        rng = np.random.default_rng(random.getrandbits(64))
        return to_tensor(synth.augment_card(self.image(item.path), rng)), item.label


class GroupBatchSampler(torch.utils.data.Sampler):
    """Fresh group-aware batches every epoch (trainset.group_batches)."""

    def __init__(self, labels: list[int], codes: list[str], batch_size: int, seed: int = 0):
        self.labels, self.codes, self.batch_size = labels, codes, batch_size
        self.rng = random.Random(seed)

    def __iter__(self):
        return iter(trainset.group_batches(self.labels, self.codes, self.batch_size, self.rng))

    def __len__(self) -> int:
        return len(trainset.group_batches(self.labels, self.codes, self.batch_size, random.Random(0)))


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def validate(model: Embedder, cards: list[Image.Image], dev: torch.device, views: int = 2) -> float:
    """Top-1 of augmented views (fixed seed) against embeddings of the clean art."""
    model.eval()
    clean = torch.stack([to_tensor(c.resize(synth.TRAIN_SIZE)) for c in cards]).to(dev)
    references = torch.cat([model(batch) for batch in clean.split(128)])
    correct = total = 0
    rng = np.random.default_rng(1234)
    for label, card in enumerate(cards):
        batch = torch.stack([to_tensor(synth.augment_card(card, rng)) for _ in range(views)]).to(dev)
        predicted = (model(batch) @ references.T).argmax(1)
        correct += int((predicted == label).sum())
        total += views
    model.train()
    return correct / total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="card-embedder")
    parser.add_argument("--scope", choices=["roster", "full"], default="full",
                        help="full = every printing with downloaded art (fetch_cards.py --art all)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--views", type=int, default=8, help="augmented views per printing per epoch")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scan-weight", type=int, default=4,
                        help="times each labeled train scan appears per epoch")
    parser.add_argument("--no-scans", action="store_true", help="art only (an ablation)")
    parser.add_argument("--parent", help="the version this one starts from (default: the shipped one, else v0)")
    parser.add_argument("--no-pretrained", action="store_true", help="random init (tests; no weight download)")
    parser.add_argument("--max-steps", type=int, help="stop early (smoke tests)")
    args = parser.parse_args(argv)

    if not remote.SAFE_NAME.fullmatch(args.name):
        raise SystemExit("--name must be letters, digits, '.', '_' or '-'")

    if args.scope == "roster":
        ids = [p["id"] for p in io.read_json(paths.DATA_CARDS / "printings.json")]
    else:
        ids = [p["printingId"] for p in io.read_json(paths.FULL_CATALOG)]
    # Negative test cards stay out of training, or the rejection threshold eval is optimistic.
    held_out = {d.name for d in paths.NEGATIVES.iterdir()} if paths.NEGATIVES.exists() else set()
    classes = [i for i in ids if (paths.ART / f"{i}.jpg").exists() and i not in held_out]
    if len(classes) < 2:
        raise SystemExit("need art for at least 2 printings; run fetch_cards.py (--art all for full)")

    dev = device()
    art_paths = {c: str(paths.ART / f"{c}.jpg") for c in classes}
    scans = [] if args.no_scans else dataset.train_records()
    items, counts = trainset.build_items(classes, art_paths, scans, args.views, args.scan_weight)
    card_ids = dataset.catalog_card_ids()
    codes = [dataset.card_id_of(c, card_ids) for c in classes]
    data = TrainingViews(items)
    loader = DataLoader(data, batch_sampler=GroupBatchSampler([i.label for i in items], codes, args.batch),
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    model = Embedder(args.dim, pretrained=not args.no_pretrained).to(dev)
    head = CosFace(args.dim, len(classes)).to(dev)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=args.lr, weight_decay=1e-4)
    total_steps = args.max_steps or args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(total_steps, 1))
    print(f"training on {counts['printings']} printings + {counts['scans']} real scans ×{args.scan_weight} "
          f"({counts['scans_skipped']} scans without art skipped), {len(items)} items/epoch, device {dev}")

    out_dir = paths.MODELS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    accuracy = 0.0
    for epoch in range(args.epochs):
        running = 0.0
        for images, labels in loader:
            loss = head(model(images.to(dev)), labels.to(dev))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            step += 1
            if step >= total_steps:
                break
        accuracy = validate(model, [data.image(art_paths[c]) for c in classes], dev)
        print(f"epoch {epoch + 1}: loss {running / max(1, len(loader)):.3f}, synthetic val top-1 {accuracy:.3f}")
        torch.save({"state_dict": model.state_dict(), "dim": args.dim, "classes": classes,
                    "input_size": synth.TRAIN_SIZE, "name": args.name}, out_dir / "checkpoint.pt")
        if step >= total_steps:
            break
    print(f"saved {out_dir / 'checkpoint.pt'}")

    parent = args.parent or ((shipped.read(paths.SHIPPED) or {}).get("name") or "v0")
    io.write_json(out_dir / "training.json", {
        "name": args.name, "created": date.today().isoformat(), "parent": parent, "classes": len(classes),
        "counts": counts, "epochs": args.epochs, "views": args.views, "batch": args.batch, "lr": args.lr,
        "dim": args.dim, "device": str(dev), "synthetic_val_top1": round(accuracy, 4)})


if __name__ == "__main__":
    main()
