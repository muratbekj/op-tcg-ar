import numpy as np
from PIL import Image

from oplab import dataset, synth


def test_perspective_coeffs_map_target_corners_to_source():
    source = np.array([[0, 0], [100, 0], [100, 140], [0, 140]], float)
    target = np.array([[10, 20], [90, 25], [95, 160], [5, 150]], float)
    a, b, c, d, e, f, g, h = synth.perspective_coeffs(source, target)
    for (x, y), (u, v) in zip(target, source):
        w = g * x + h * y + 1
        assert abs((a * x + b * y + c) / w - u) < 1e-6
        assert abs((d * x + e * y + f) / w - v) < 1e-6


def test_render_photo_and_augment_sizes():
    card = Image.new("RGB", (315, 440), (200, 30, 30))
    rng = np.random.default_rng(0)
    for condition in synth.CONDITIONS:
        photo, tags = synth.render_photo(card, rng, condition)
        assert photo.size == synth.CANVAS and tags["condition"] == condition
    assert synth.augment_card(card, rng).size == synth.TRAIN_SIZE


def test_card_stays_in_frame():
    for seed in range(200):
        rng = np.random.default_rng(seed)
        corners = synth.card_corners(rng, synth.CONDITIONS[seed % len(synth.CONDITIONS)], 600 / 838)
        assert corners.min() >= 0
        assert (corners.max(0) <= synth.CANVAS).all()
