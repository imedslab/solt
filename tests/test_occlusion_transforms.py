import random

import cv2
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

import solt.core as slc
import solt.transforms as slt
import solt.utils as slu

from .fixtures import *


@pytest.mark.parametrize(
    "img, expected, cut_size",
    [
        (lf("img_7x7"), np.zeros((7, 7, 1), dtype=np.uint8), 7),
        (lf("img_6x6"), np.zeros((6, 6, 1), dtype=np.uint8), 6),
        (lf("img_6x6_rgb"), np.zeros((6, 6, 3), dtype=np.uint8), 6),
        (lf("img_7x7"), np.zeros((7, 7, 1), dtype=np.uint8), 1.0),
        (lf("img_6x6"), np.zeros((6, 6, 1), dtype=np.uint8), 1.0),
        (lf("img_6x6_rgb"), np.zeros((6, 6, 3), dtype=np.uint8), 1.0),
    ],
)
def test_cutout_blacks_out_image(img, expected, cut_size):
    dc = slc.DataContainer((img,), "I")
    trf = slc.Stream([slt.CutOut(p=1, cutout_size=cut_size)])

    dc_res = trf(dc, return_torch=False)

    assert np.array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("cut_size", [1, (1, 1), 0.5, (0.5, 0.5)])
def test_cutout_1x1_blacks_corner_pixels_2x2_img(img_2x2, cut_size):
    dc = slc.DataContainer((img_2x2.copy(),), "I")
    trf = slc.Stream([slt.CutOut(p=1, cutout_size=cut_size)])
    dc_res = trf(dc, return_torch=False)

    equal = 0
    for i in range(2):
        for j in range(2):
            tmp_opt = img_2x2.copy()
            tmp_opt[i, j] = 0
            if np.array_equal(dc_res.data[0], tmp_opt):
                equal += 1

    assert equal == 1


@pytest.mark.parametrize(
    "img, d_range, rotate, ratio, expected",
    # With this seed, the 3x3 image is inside a single dropped square, apart from one kept vertical line in the middle
    [(np.ones((3, 3, 3)), (5, 10), 1, 0.2, np.broadcast_to(np.array([0.0, 1.0, 0.0])[None, :, None], (3, 3, 3)))],
)
def test_gridmask_transform(img, d_range, rotate, ratio, expected):
    np.random.seed(42)
    random.seed(42)
    dc = slc.DataContainer((img,), "I")
    trf = slc.Stream([slt.GridMask(d_range=d_range, rotate=rotate, ratio=ratio, p=1)])

    dc_res = trf(dc, return_torch=False)
    print(dc_res.data[0])
    assert np.array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("d_range", [(-1, 0)])
def test_gridmask_d_range_raises_error_when_wrong(d_range):
    with pytest.raises(ValueError):
        slt.GridMask(d_range=d_range, p=1)


@pytest.mark.parametrize("rotate", [0.1])
def test_gridmask_rotate_raises_error_when_wrong(rotate):
    with pytest.raises(TypeError):
        slt.GridMask(rotate=rotate, p=1)


@pytest.mark.parametrize("rotate", [(0.1, 0.1)])
def test_gridmask_rotate_raises_error_when_wrong_tuple(rotate):
    with pytest.raises(TypeError):
        slt.GridMask(rotate=rotate, p=1)


def _cutout_rectangle(img):
    """Width and height of the single blacked out rectangle in an image of ones."""
    ys, xs = np.nonzero(img[..., 0] == 0)
    return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


@pytest.mark.parametrize(
    "cutout_size, w_range, h_range",
    [
        (((2, 6), (3, 5)), (2, 6), (3, 5)),
        (((0.1, 0.3), (0.2, 0.4)), (4, 12), (8, 16)),
        ((4, (3, 5)), (4, 4), (3, 5)),
    ],
)
def test_cutout_random_size(cutout_size, w_range, h_range):
    random.seed(0)
    sizes = set()
    for _ in range(100):
        img = np.ones((40, 40, 1), dtype=np.uint8)
        res = slt.CutOut(cutout_size=cutout_size, p=1)(slc.DataContainer((img,), "I")).data[0]
        w, h = _cutout_rectangle(res)
        assert w_range[0] <= w <= w_range[1]
        assert h_range[0] <= h <= h_range[1]
        sizes.add((w, h))
    assert len(sizes) > 1


def test_cutout_sizes_are_sampled_per_cut():
    random.seed(0)
    trf = slt.CutOut(cutout_size=((2, 10), (2, 10)), n_cuts=5, p=1)
    trf(slc.DataContainer((np.ones((40, 40, 1), dtype=np.uint8),), "I"))
    sizes = list(zip(trf.state_dict["cut_size_x"], trf.state_dict["cut_size_y"]))
    assert len(sizes) == 5
    assert len(set(sizes)) > 1


def test_cutout_random_number_of_cuts():
    random.seed(0)
    n_cuts = set()
    for _ in range(50):
        trf = slt.CutOut(cutout_size=2, n_cuts=(1, 4), p=1)
        trf(slc.DataContainer((np.ones((40, 40, 1), dtype=np.uint8),), "I"))
        assert 1 <= len(trf.state_dict["x"]) <= 4
        n_cuts.add(len(trf.state_dict["x"]))
    assert n_cuts == {1, 2, 3, 4}


def test_cutout_fixed_size_keeps_the_random_sequence():
    # A fixed size must not draw random numbers, so that seeded pipelines give the same results as before
    random.seed(0)
    trf = slt.CutOut(cutout_size=(3, 4), n_cuts=2, p=1)
    trf(slc.DataContainer((np.ones((20, 20, 1), dtype=np.uint8),), "I"))
    random.seed(0)
    random.random()  # Whether to apply the transform
    expected = [
        int(random.random() * 17),
        int(random.random() * 16),
        int(random.random() * 17),
        int(random.random() * 16),
    ]
    xy = [v for pair in zip(trf.state_dict["x"], trf.state_dict["y"]) for v in pair]
    assert xy == expected


@pytest.mark.parametrize(
    "cutout_size, n_cuts, error",
    [
        (((6, 2), 2), 1, ValueError),
        (((-1, 2), 2), 1, ValueError),
        (((2, 4), 0.5), 1, TypeError),
        (((2, 4.5), 2), 1, TypeError),
        (((2, 4, 6), 2), 1, ValueError),
        ((2, 2, 2), 1, ValueError),
        (2, (3, 1), ValueError),
        (2, (-1, 2), ValueError),
        (2, (1.0, 2), TypeError),
        (2, 1.5, TypeError),
    ],
)
def test_cutout_wrong_random_size_or_n_cuts(cutout_size, n_cuts, error):
    with pytest.raises(error):
        slt.CutOut(cutout_size=cutout_size, n_cuts=n_cuts)


def test_cutout_random_size_too_large():
    trf = slt.CutOut(cutout_size=((2, 50), 2), p=1)
    with pytest.raises(ValueError):
        trf(slc.DataContainer((np.ones((20, 20, 1), dtype=np.uint8),), "I"))


def test_cutout_random_size_serialization():
    trf = slt.CutOut(cutout_size=((2, 6), (0, 4)), n_cuts=(1, 3))
    restored = slu.from_yaml(slc.Stream([trf]).to_yaml()).transforms[0]
    assert tuple(map(tuple, restored.cutout_size)) == ((2, 6), (0, 4))
    assert tuple(restored.n_cuts) == (1, 3)


# GridMask. Without rotation, the grid is made of kept lines of width ``l = ceil(d * ratio)`` repeated with the
# period ``d``, and the dropped region is the squares of side ``s = d - l`` between them.
def _gridmask_line_and_square(d, ratio):
    line = int(np.ceil(np.round(d * ratio, 6)))
    return line, d - line


def _gridmask_dropped(shape, seed, **kwargs):
    random.seed(seed)
    img = np.ones((*shape, 1), dtype=np.float32)
    res = slt.GridMask(p=1, **kwargs)(slc.DataContainer((img,), "I"), return_torch=False).data[0]
    assert set(np.unique(res)).issubset({0, 1})
    return res[:, :, 0] == 0


def _runs(vec):
    """Lengths of the runs of ``True`` and ``False`` in ``vec``, without the runs touching the borders."""
    edges = np.flatnonzero(np.diff(vec.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [vec.size]))
    runs = [(bool(vec[s]), e - s) for s, e in zip(bounds[:-1], bounds[1:])]
    return runs[1:-1]


@pytest.fixture(params=[4, 10, 16], ids=lambda d: f"d{d}")
def gridmask_d(request):
    return request.param


@pytest.fixture(params=[0.3, 0.5, 0.7], ids=lambda r: f"ratio{r}")
def gridmask_ratio(request):
    return request.param


@pytest.fixture(params=[(64, 64), (48, 80), (37, 53)], ids=["square", "non_square", "odd"])
def gridmask_shape(request):
    return request.param


@pytest.fixture(params=["crop", "none", None])
def gridmask_default_mode(request):
    return request.param


@pytest.fixture(params=["reverse", "reserve"])
def gridmask_inverse_mode(request):
    return request.param


@pytest.fixture(
    params=[0, None, 15, (30, 45), (-20, -10)], ids=["no_rotation", "none", "int", "tuple", "negative_tuple"]
)
def gridmask_rotate(request):
    return request.param


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gridmask_drops_squares_between_kept_lines(
    gridmask_d, gridmask_ratio, gridmask_shape, gridmask_default_mode, seed
):
    d, (h, w) = gridmask_d, gridmask_shape
    line, square = _gridmask_line_and_square(d, gridmask_ratio)
    dropped = _gridmask_dropped(gridmask_shape, seed, d_range=(d, d), ratio=gridmask_ratio, mode=gridmask_default_mode)

    rows, cols = dropped.any(axis=1), dropped.any(axis=0)
    # The dropped region is a union of squares: a pixel is dropped iff both its row and its column are between lines
    np.testing.assert_array_equal(dropped, np.outer(rows, cols))
    for vec in (rows, cols):
        # Any d consecutive rows (columns) contain exactly one side of a square, also at the image borders
        windows = np.lib.stride_tricks.sliding_window_view(vec, d).sum(axis=1)
        np.testing.assert_array_equal(windows, square)
        for is_dropped, length in _runs(vec):
            assert length == (square if is_dropped else line)

    assert abs(dropped.mean() - (square / d) ** 2) <= 2 * square / min(h, w)


def test_gridmask_dropped_fraction_is_one_minus_ratio_squared():
    dropped = _gridmask_dropped((256, 256), 0, d_range=(16, 16), ratio=0.5, mode="crop")
    assert dropped.mean() == pytest.approx(0.25, abs=0.02)


def test_gridmask_inverse_mode_drops_the_complement(
    gridmask_d, gridmask_ratio, gridmask_shape, gridmask_inverse_mode, gridmask_rotate
):
    kwargs = {"d_range": (gridmask_d, gridmask_d), "ratio": gridmask_ratio, "rotate": gridmask_rotate}
    dropped = _gridmask_dropped(gridmask_shape, 0, mode="crop", **kwargs)
    dropped_inverse = _gridmask_dropped(gridmask_shape, 0, mode=gridmask_inverse_mode, **kwargs)
    np.testing.assert_array_equal(dropped_inverse, ~dropped)


@pytest.mark.parametrize("modes", [("crop", "none"), ("crop", None), ("reverse", "reserve")])
def test_gridmask_mode_aliases_are_identical(modes, gridmask_rotate):
    kwargs = {"d_range": (5, 20), "ratio": 0.4, "rotate": gridmask_rotate}
    res = [_gridmask_dropped((50, 70), 7, mode=mode, **kwargs) for mode in modes]
    np.testing.assert_array_equal(res[0], res[1])


def test_gridmask_rotation_angle_is_sampled_in_range(gridmask_rotate, gridmask_default_mode, monkeypatch):
    angles = []
    get_rotation_matrix = cv2.getRotationMatrix2D

    def recording_get_rotation_matrix(center, angle, scale):
        angles.append(angle)
        return get_rotation_matrix(center, angle, scale)

    monkeypatch.setattr(cv2, "getRotationMatrix2D", recording_get_rotation_matrix)
    for seed in range(20):
        dropped = _gridmask_dropped(
            (40, 60), seed, d_range=(8, 12), ratio=0.5, rotate=gridmask_rotate, mode=gridmask_default_mode
        )
        # Every unit of the grid drops about (1 - ratio) ** 2 of its area, whatever the angle
        assert dropped.mean() == pytest.approx(0.25, abs=0.12)

    if gridmask_rotate in (0, None):
        assert angles == []
    else:
        low, high = (-gridmask_rotate, gridmask_rotate) if isinstance(gridmask_rotate, int) else gridmask_rotate
        assert len(angles) == 20
        assert all(low <= angle <= high for angle in angles)


@pytest.mark.parametrize(
    "rotate, expected", [(0, 0), (None, 0), (15, (-15, 15)), ((30, 45), (30, 45)), ((-5, -5), (-5, -5))]
)
def test_gridmask_rotate_range(rotate, expected):
    assert slt.GridMask(rotate=rotate).rotate == expected


@pytest.mark.parametrize("rotate", [-10, (1, 2, 3), (10,), (45, 30)])
def test_gridmask_rotate_raises_value_error_when_wrong(rotate):
    with pytest.raises(ValueError):
        slt.GridMask(rotate=rotate)


@pytest.mark.parametrize("ratio", [-0.1, 1.5])
def test_gridmask_ratio_raises_error_when_wrong(ratio):
    with pytest.raises(ValueError):
        slt.GridMask(ratio=ratio)


@pytest.mark.parametrize("mode", ["inverse", "reversed"])
def test_gridmask_mode_raises_error_when_wrong(mode):
    with pytest.raises(ValueError):
        slt.GridMask(mode=mode)


@pytest.mark.parametrize("n_channels", [1, 3, 5])
def test_gridmask_applies_the_same_mask_to_every_channel(n_channels, gridmask_rotate, gridmask_inverse_mode):
    random.seed(3)
    img = np.random.randint(1, 255, (45, 61, n_channels), dtype=np.uint8)
    img_orig = img.copy()
    trf = slt.GridMask(d_range=(6, 15), ratio=0.5, rotate=gridmask_rotate, mode=gridmask_inverse_mode, p=1)
    res = trf(slc.DataContainer((img,), "I"), return_torch=False).data[0]

    dropped = res == 0
    assert dropped.any() and not dropped.all()
    for k in range(n_channels):
        np.testing.assert_array_equal(dropped[:, :, k], dropped[:, :, 0])
    np.testing.assert_array_equal(res[~dropped], img_orig[~dropped])


def test_gridmask_supports_2d_images():
    random.seed(0)
    img = np.ones((40, 50), dtype=np.uint8)
    res = slt.GridMask(d_range=(10, 10), ratio=0.5, p=1)(slc.DataContainer((img,), "I"), return_torch=False).data[0]
    assert res.shape == (40, 50, 1)
    assert (res == 0).mean() == pytest.approx(0.25, abs=0.1)


def test_gridmask_modifies_the_image_in_place():
    img = np.ones((32, 32, 3), dtype=np.uint8)
    res = slt.GridMask(d_range=(8, 8), p=1)(slc.DataContainer((img,), "I"), return_torch=False).data[0]
    assert res is img
    assert (img == 0).any()


@pytest.mark.parametrize("mode", ["crop", "reverse"])
def test_gridmask_does_not_change_masks_keypoints_distance_maps_boxes_and_labels(mode, gridmask_rotate):
    h, w = 41, 51
    rng = np.random.default_rng(0)
    img = rng.integers(1, 255, (h, w, 3), dtype=np.uint8)
    mask = (rng.random((h, w)) > 0.5).astype(np.uint8)
    dmap = rng.random((h, w)).astype(np.float32)
    kpts = slc.Keypoints(np.array([[25.0, 10.0], [3.0, 37.0]]), h, w)
    boxes = slc.BoundingBoxes([[10.0, 5.0, 30.0, 20.0]], h, w)
    data = (img, mask.copy(), dmap.copy(), slc.Keypoints(kpts.data.copy(), h, w), boxes, 1)

    trf = slt.GridMask(d_range=(5, 10), rotate=gridmask_rotate, mode=mode, p=1)
    res = trf(slc.DataContainer(data, "IMDPBL"), return_torch=False).data
    assert (res[0] == 0).any()
    np.testing.assert_array_equal(res[1], mask)
    np.testing.assert_array_equal(res[2], dmap)
    np.testing.assert_array_equal(res[3].data, kpts.data)
    np.testing.assert_array_equal(res[4].data, [[10.0, 5.0, 30.0, 20.0]])
    assert res[5] == 1
