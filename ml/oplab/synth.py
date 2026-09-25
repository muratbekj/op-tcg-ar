"""Synthetic card photos: a card warped onto a desk-like background with realistic nuisances.

Used to build a benchmark before you have real photos, and as training augmentation. Real photos of
physical cards remain the test set that matters: synthetic images are made from the same (SAMPLE-
watermarked) API art as the references, so they overestimate accuracy on real cards.
"""

import io as _io

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

CONDITIONS = ["clean", "angle", "glare", "dim", "blur", "upside_down", "small"]
CANVAS = (900, 1200)  # portrait, like the phone camera
TRAIN_SIZE = (224, 320)  # width, height: the Core ML embedder input


def perspective_coeffs(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Coefficients for PIL's PERSPECTIVE transform, which maps output (target) points back to input (source)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(target, source):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    return np.linalg.solve(np.asarray(rows, float), np.asarray(rhs, float))


def warp(image: Image.Image, corners: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Warps `image` so its corners (TL, TR, BR, BL) land on `corners` in a transparent canvas."""
    w, h = image.size
    source = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    coeffs = perspective_coeffs(source, corners)
    return image.convert("RGBA").transform(size, Image.Transform.PERSPECTIVE, tuple(coeffs), Image.Resampling.BICUBIC)


def card_corners(rng: np.random.Generator, condition: str, card_aspect: float) -> np.ndarray:
    width, height = CANVAS
    card_w = width * (rng.uniform(0.28, 0.36) if condition == "small" else rng.uniform(0.45, 0.68))
    card_h = card_w / card_aspect
    local = np.array([[-card_w / 2, -card_h / 2], [card_w / 2, -card_h / 2], [card_w / 2, card_h / 2], [-card_w / 2, card_h / 2]])

    if condition == "angle":
        # Looking at the card from its bottom edge: the far (top) edge shrinks and the card foreshortens.
        shrink = rng.uniform(0.62, 0.8)
        local[0, 0] *= shrink
        local[1, 0] *= shrink
        local[:, 1] *= rng.uniform(0.7, 0.85)

    theta = np.deg2rad(rng.uniform(-12, 12) + (180 if condition == "upside_down" else 0))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = local @ rotation.T
    # Keep the whole card in frame with a small margin; a card cut off by the frame edge isn't a
    # recognition failure worth measuring.
    margin = 0.03 * np.array([width, height])
    low = margin - corners.min(0)
    high = np.array([width, height]) - margin - corners.max(0)
    center = np.array([width / 2, height / 2]) + rng.uniform(-0.12, 0.12, 2) * [width, height]
    return corners + np.clip(center, low, np.maximum(low, high))


def background(rng: np.random.Generator) -> Image.Image:
    width, height = CANVAS
    base = rng.uniform(0.05, 0.75, 3)
    tint = base + rng.uniform(-0.12, 0.12, 3)
    ramp = np.linspace(0, 1, height)[:, None, None]
    image = (base * (1 - ramp) + tint * ramp) * np.ones((height, width, 3))
    grain = rng.normal(0, 1, (height // 8, width // 8, 1))
    grain = np.array(Image.fromarray(((grain - grain.min()) / (np.ptp(grain) + 1e-6) * 255).astype(np.uint8)[:, :, 0])
                     .resize((width, height), Image.Resampling.BICUBIC), float)[:, :, None] / 255
    image = np.clip(image + (grain - 0.5) * rng.uniform(0.02, 0.1), 0, 1)
    return Image.fromarray((image * 255).astype(np.uint8))


def add_glare(image: Image.Image, rng: np.random.Generator, region: tuple[float, float, float, float]) -> Image.Image:
    x0, y0, x1, y1 = map(float, region)
    glare = Image.new("L", image.size, 0)
    cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
    rx, ry = rng.uniform(0.15, 0.35) * (x1 - x0), rng.uniform(0.08, 0.2) * (y1 - y0)
    ImageDraw.Draw(glare).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(rng.uniform(140, 230)))
    glare = glare.filter(ImageFilter.GaussianBlur(float(max(rx, ry)) * 0.35))
    return Image.composite(Image.new("RGB", image.size, (255, 255, 250)), image, glare)


def finish(image: Image.Image, rng: np.random.Generator, noise: float = 4.0, quality: tuple[int, int] = (70, 92)) -> Image.Image:
    """Sensor noise and JPEG compression, applied to everything."""
    pixels = np.asarray(image, float) + rng.normal(0, noise, (image.size[1], image.size[0], 3))
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    buffer = _io.BytesIO()
    image.save(buffer, "JPEG", quality=int(rng.integers(*quality)))
    return Image.open(_io.BytesIO(buffer.getvalue())).convert("RGB")


def render_photo(card: Image.Image, rng: np.random.Generator, condition: str | None = None) -> tuple[Image.Image, dict]:
    """A synthetic phone photo of `card`. Returns the image and its tags."""
    condition = condition or str(rng.choice(CONDITIONS))
    card = card.convert("RGB")
    corners = card_corners(rng, condition, card.size[0] / card.size[1])

    photo = background(rng)
    warped = warp(card, corners, CANVAS)
    photo.paste(warped, (0, 0), warped)

    if condition == "glare":
        photo = add_glare(photo, rng, (*corners.min(0), *corners.max(0)))
    photo = ImageEnhance.Brightness(photo).enhance(float(rng.uniform(0.4, 0.65) if condition == "dim" else rng.uniform(0.85, 1.15)))
    photo = ImageEnhance.Contrast(photo).enhance(float(rng.uniform(0.85, 1.15)))
    if condition == "blur":
        photo = photo.filter(ImageFilter.GaussianBlur(float(rng.uniform(1.5, 3.0))))
    photo = finish(photo, rng, noise=8.0 if condition == "dim" else 4.0)
    return photo, {"condition": condition, "corners": corners.round(1).tolist()}


def augment_card(card: Image.Image, rng: np.random.Generator, size: tuple[int, int] = TRAIN_SIZE) -> Image.Image:
    """A training view of `card` as the device would see it after rectification: small residual
    perspective error, lighting changes, glare, blur, noise. Half the time the watermark band
    (where API art says SAMPLE) is blurred out so the model can't lean on it."""
    card = card.convert("RGB")
    w, h = card.size
    jitter = rng.uniform(-0.04, 0.04, (4, 2)) * [w, h]
    source = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) + jitter
    target = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    view = card.transform((w, h), Image.Transform.PERSPECTIVE, tuple(perspective_coeffs(source, target)), Image.Resampling.BICUBIC)

    if rng.random() < 0.5:
        top, bottom = int(h * 0.38), int(h * 0.6)
        band = view.crop((0, top, w, bottom)).filter(ImageFilter.GaussianBlur(12))
        view.paste(band, (0, top))
    if rng.random() < 0.35:
        view = add_glare(view, rng, (0, 0, w, h))
    view = ImageEnhance.Brightness(view).enhance(float(rng.uniform(0.55, 1.3)))
    view = ImageEnhance.Contrast(view).enhance(float(rng.uniform(0.75, 1.25)))
    view = ImageEnhance.Color(view).enhance(float(rng.uniform(0.7, 1.3)))
    if rng.random() < 0.3:
        view = view.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.0))))
    view = view.resize(size, Image.Resampling.BILINEAR)
    return finish(view, rng, noise=rng.uniform(0, 6), quality=(60, 95))
