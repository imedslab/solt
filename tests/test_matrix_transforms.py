import copy

import cv2
import numpy as np
import pytest

import solt.core as slc
import solt.transforms as slt
from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS

from .fixtures import *


def test_rotate_range_none():
    trf = slt.Rotate(None)
    assert trf.angle_range == (0, 0)


@pytest.mark.parametrize("angle", [1, 2.5])
def test_rotate_range_conversion_from_number(angle):
    trf = slt.Rotate(angle_range=angle)
    assert trf.angle_range == (-angle, angle)


def test_shear_range_none():
    trf = slt.Shear(None, None)
    assert trf.range_x == (0, 0)
    assert trf.range_y == (0, 0)


@pytest.mark.parametrize("ignore_state", [True, False])
@pytest.mark.parametrize(
    "transform_settings",
    [
        None,
        {0: {"interpolation": "nearest", "padding": "z"}},
        {0: {"interpolation": "nearest", "padding": "r"}},
        {0: {"interpolation": "bilinear", "padding": "z"}},
        {0: {"interpolation": "bilinear", "padding": "r"}},
        {0: {"interpolation": "bicubic", "padding": "z"}},
        {0: {"interpolation": "bicubic", "padding": "r"}},
        {0: {"interpolation": "area", "padding": "z"}},
        {0: {"interpolation": "area", "padding": "r"}},
        {0: {"interpolation": "lanczos", "padding": "z"}},
        {0: {"interpolation": "lanczos", "padding": "r"}},
    ],
)
def test_rotate_90_img_mask_keypoints_destructive(img_3x3, mask_3x3, transform_settings, ignore_state):
    # Setting up the data

    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3
    H, W = mask.shape

    dc = slc.DataContainer(
        (img, mask, kpts, 1),
        "IMPL",
        transform_settings=copy.deepcopy(transform_settings),
    )
    # Defining the 90 degrees transform (clockwise)
    stream = slt.Rotate(angle_range=(90, 90), p=1, ignore_state=ignore_state)
    dc_res = stream(dc)

    img_res, _, _ = dc_res[0]
    mask_res, _, _ = dc_res[1]
    kpts_res, _, _ = dc_res[2]
    label_res, _, _ = dc_res[3]
    M = cv2.getRotationMatrix2D((W // 2, H // 2), -90, 1)

    img_inter = ALLOWED_INTERPOLATIONS["bicubic"]
    img_pad = ALLOWED_PADDINGS["z"]
    if transform_settings is not None:
        img_inter = ALLOWED_INTERPOLATIONS[transform_settings[0]["interpolation"]]
        img_pad = ALLOWED_PADDINGS[transform_settings[0]["padding"]]

    expected_img_res = cv2.warpAffine(img, M, (W, H), flags=img_inter, borderMode=img_pad).reshape((H, W, 1))
    expected_mask_res = cv2.warpAffine(mask, M, (W, H))
    expected_kpts_res = np.array([[2, 0], [0, 0], [0, 2], [2, 2]]).reshape((4, 2))

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)
    np.testing.assert_array_almost_equal(expected_kpts_res, kpts_res.data)
    assert label_res == 1


@pytest.mark.parametrize("k", list(range(-4, 5)))
def test_rotate_90_img_mask_nondestructive(k, img_3x3, mask_3x3):
    # Setting up the data
    img, mask = img_3x3, mask_3x3
    H, W = mask.shape

    dc = slc.DataContainer((img, mask), "IM")
    # Defining the 90 degrees transform (counterclockwise)
    stream = slt.Rotate90(k=k, p=1)
    dc_res = stream(dc)

    img_res, _, _ = dc_res[0]
    mask_res, _, _ = dc_res[1]

    expected_img_res = np.rot90(img, -k).reshape((H, W, 1))
    expected_mask_res = np.rot90(mask, -k)

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)


@pytest.mark.parametrize("k", [None, "123", 123.0])
def test_rotate_nondestructive_does_not_accept_non_int_k(k):
    with pytest.raises(TypeError):
        slt.Rotate90(k=k)


@pytest.mark.parametrize("k", list(range(-4, 5)))
def test_rotate_90_transforms_have_same_behaviour(k, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf_1 = slt.Rotate(angle_range=(k * 90, k * 90), p=1)
    trf_1.sample_transform(dc)

    trf_2 = slt.Rotate90(k=k, p=1)
    trf_2.sample_transform(dc)

    assert np.array_equal(trf_1.state_dict["transform_matrix"], trf_2.state_dict["transform_matrix"])


def test_zoom_x_axis_odd(img_5x5):
    stream = slt.Scale(range_x=(0.5, 0.5), range_y=(1, 1), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_5x5,), "I")
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_x_axis_even(img_6x6):
    stream = slt.Scale((0.5, 0.5), (1, 1), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_6x6,), "I")
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_odd(img_5x5):
    stream = slt.Scale((0.5, 0.5), (3, 3), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_5x5,), "I")
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 3 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_even(img_6x6):
    stream = slt.Scale((0.5, 0.5), (2, 2), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img_6x6,), "I")
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 2 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_img_mask(img_3x4, mask_3x4):
    img_mask_3x4 = img_3x4, mask_3x4
    stream = slt.Scale((0.5, 0.5), (2, 2), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer(img_mask_3x4, "IM")
    H, W = img_mask_3x4[0].shape[0], img_mask_3x4[0].shape[1]
    img_res = stream(dc)[0][0]
    mask_res = stream(dc)[1][0]
    assert H * 2 == img_res.shape[0], W // 2 == img_res.shape[1]
    assert H * 2 == mask_res.shape[0], W // 2 == mask_res.shape[1]


def test_keypoints_assert_reflective(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = slc.DataContainer(
        (
            img,
            mask,
            kpts,
        ),
        "IMP",
    )
    # Defining the 90 degrees transform (clockwise)
    stream = slt.Rotate(angle_range=(20, 20), p=1, padding="r")
    with pytest.raises(ValueError):
        stream(dc)


def test_translate_forward_backward_sampling(img_6x6_rgb):
    stream = slc.Stream(
        [
            slt.Translate(range_x=(1, 1), range_y=(1, 1), p=1),
            slt.Translate(range_x=(-1, -1), range_y=(-1, -1), p=1),
        ]
    )
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = stream.optimize_transforms_stack(stream.transforms, dc)[0]
    assert 1 == trf.state_dict["translate_x"]  # The settings will be overrided by the first transform
    assert 1 == trf.state_dict["translate_y"]  # The settings will be overrided by the first transform
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_empty_sampling_no_trf(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(affine_transforms=slc.Stream(), p=1)
    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_empty_sampling_low_prob_trf(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(affine_transforms=slc.Stream([slt.Rotate(p=0)]), p=1)

    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_empty_sampling_from_many_low_prob_trf(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(
        affine_transforms=slc.Stream(
            [
                slt.Rotate(angle_range=(70, 70), p=0),
                slt.Rotate(angle_range=(10, 10), p=0),
                slt.Rotate(angle_range=(-2, -2), p=0),
            ]
        ),
        p=1,
    )
    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_projection_translate_plus_minus_1(img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Projection(
        affine_transforms=slc.Stream(
            [
                slt.Translate(range_x=(1, 1), range_y=(1, 1), p=1),
                slt.Translate(range_x=(-1, -1), range_y=(-1, -1), p=1),
            ]
        ),
        p=1,
    )

    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


@pytest.mark.parametrize("scale,expected", [(2, (1, 2)), (2.5, (1, 2.5)), (0.5, (0.5, 1)), (-1, None)])
def test_scale_range_from_number(scale, expected):
    if expected is not None:
        trf = slt.Scale(range_x=scale, range_y=scale)
        assert trf.range_x == expected
        assert trf.range_x == expected
    else:
        with pytest.raises(ValueError):
            slt.Scale(range_x=scale)
        with pytest.raises(ValueError):
            slt.Scale(range_x=None, range_y=scale)


@pytest.mark.parametrize(
    "same, scale_x, scale_y, expected",
    [
        (True, (2, 2), (2, 2), (2, 2)),
        (True, (2, 2), (1, 1), (2, 2)),
        (True, (2, 2), None, (2, 2)),
        (True, None, (2, 2), (2, 2)),
        (False, (2, 2), (2, 2), (2, 2)),
        (False, (2, 2), (3, 3), (2, 3)),
        (False, (2, 2), None, (2, 1)),
        (False, None, (2, 2), (1, 2)),
    ],
)
def test_scale_sampling_scale(same, scale_x, scale_y, expected, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Scale(range_x=scale_x, range_y=scale_y, same=same)
    trf.sample_transform(dc)
    assert expected == (trf.state_dict["scale_x"], trf.state_dict["scale_y"])


@pytest.mark.parametrize(
    "translate,expected",
    [
        (2, (-2, 2)),
        (2.5, (-2.5, 2.5)),
        (0.5, (-0.5, 0.5)),
        (-0.5, (-0.5, 0.5)),
    ],
)
def test_translate_range_from_number(translate, expected):
    trf = slt.Translate(range_x=translate, range_y=translate)
    assert trf.range_x == expected
    assert trf.range_y == expected


@pytest.mark.parametrize(
    "scale_x, scale_y, same, expected",
    [
        (None, (2, 2), True, (2, 2)),
        ((2, 2), None, True, (2, 2)),
        (None, None, True, (1, 1)),
        (None, (2, 2), False, (1, 2)),
        ((2, 2), None, False, (2, 1)),
        (None, None, False, (1, 1)),
    ],
)
def test_scale_when_range_x_is_none(scale_x, scale_y, same, expected, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Scale(range_x=scale_x, range_y=scale_y, same=same, p=1)
    trf.sample_transform(dc)
    assert (trf.state_dict["scale_x"], trf.state_dict["scale_y"]) == expected


@pytest.mark.parametrize(
    "translate_x, translate_y, expected",
    [
        (None, (2, 2), (0, 2)),
        ((2, 2), None, (2, 0)),
        (None, None, (0, 0)),
    ],
)
def test_translate_when_range_x_is_none(translate_x, translate_y, expected, img_6x6_rgb):
    dc = slc.DataContainer(img_6x6_rgb, "I")
    trf = slt.Translate(range_x=translate_x, range_y=translate_y, p=1)
    trf.sample_transform(dc)
    assert (trf.state_dict["translate_x"], trf.state_dict["translate_y"]) == expected


@pytest.mark.parametrize(
    "param_set",
    [
        {"affine_transforms": "123"},
        {"affine_transforms": 123},
        {"affine_transforms": []},
        {
            "affine_transforms": slc.Stream(
                [
                    slt.Flip(),
                ]
            )
        },
        {
            "affine_transforms": slc.Stream(
                [
                    slt.Flip(),
                ]
            )
        },
        {"v_range": "123"},
        {"v_range": 123},
        {"v_range": ("123", "456")},
        {"v_range": ((2,), (4,))},
    ],
)
def test_random_projection_raises_type_errors(param_set):
    with pytest.raises(TypeError):
        slt.Projection(**param_set)


def test_random_proj_and_selective_stream(img_5x5):
    img = img_5x5
    dc = slc.DataContainer((img,), "I")

    ppl = slt.Projection(
        slc.SelectiveStream(
            [
                slt.Rotate(angle_range=(90, 90), p=0),
                slt.Scale(range_y=(0, 0.1), same=True, p=0),
                slt.Shear(range_y=(-0.1, 0.1), p=0),
            ],
            n=3,
        ),
        v_range=(0, 0),
    )

    dc_res = ppl(dc)

    assert np.array_equal(dc.data, dc_res.data)


@pytest.mark.parametrize("h, w", [(60, 80), (61, 81), (60, 60), (61, 61), (7, 4)])
@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_rotate_90_keypoints_follow_the_image(h, w, k):
    rng = np.random.default_rng(0)
    xy = np.stack([rng.integers(0, w, 5), rng.integers(0, h, 5)], axis=1)
    for x, y in xy:
        img = np.zeros((h, w), dtype=np.uint8)
        img[y, x] = 1
        kpts = slc.Keypoints(np.array([[float(x), float(y)]]), h, w)
        img_res, kpts_res = slt.Rotate90(k=k, p=1)(slc.DataContainer((img, kpts), "MP")).data
        ys, xs = np.nonzero(img_res)
        np.testing.assert_array_equal(kpts_res.data, [[xs[0], ys[0]]])
        assert (kpts_res.height, kpts_res.width) == img_res.shape


@pytest.mark.parametrize("rotate_first", [True, False])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotate_90_in_optimized_stream(rotate_first, k):
    h, w = 41, 51
    img = np.zeros((h, w), dtype=np.uint8)
    img[10, 25] = 1
    kpts = slc.Keypoints(np.array([[25.0, 10.0]]), h, w)

    def make_transforms():
        rotate = slt.Rotate90(k=k, p=1)
        translate = slt.Translate(range_x=(5, 5), range_y=(-3, -3), p=1)
        return [rotate, translate] if rotate_first else [translate, rotate]

    expected = slc.Stream(make_transforms())(slc.DataContainer((img, kpts), "MP"), return_torch=False)
    res = slc.Stream(make_transforms(), optimize_stack=True)(slc.DataContainer((img, kpts), "MP"), return_torch=False)

    np.testing.assert_array_equal(np.argwhere(res.data[0] == 1), np.argwhere(expected.data[0] == 1))
    np.testing.assert_allclose(res.data[1].data, expected.data[1].data, atol=1e-6)


def _rotate90_data(fmt):
    h, w = 41, 51
    rng = np.random.default_rng(0)
    if fmt == "I":
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    if fmt == "M":
        return (rng.random((h, w)) > 0.5).astype(np.uint8)
    if fmt == "D":
        return rng.random((h, w)).astype(np.float32)
    if fmt == "P":
        return slc.Keypoints(np.array([[25.0, 10.0], [3.0, 37.0]]), h, w)
    return slc.BoundingBoxes([[10.0, 5.0, 30.0, 20.0]], h, w)


@pytest.fixture(params=["I", "M", "D", "P", "B"], ids=["image", "mask", "distance_map", "keypoints", "bboxes"])
def rotate90_item(request):
    return request.param, _rotate90_data(request.param)


@pytest.fixture(params=[1, 2, 3], ids=["k1", "k2", "k3"])
def rotate90_k(request):
    return request.param


def _as_array(item):
    return item.data if isinstance(item, (slc.Keypoints, slc.BoundingBoxes)) else item


def test_fused_rotate_90_transforms_all_data_types(rotate90_item, rotate90_k):
    fmt, item = rotate90_item

    def make_transforms():
        return [slt.Rotate90(k=rotate90_k, p=1), slt.Translate(range_x=(5, 5), range_y=(-3, -3), p=1)]

    # max_bbox_* are disabled, so that the boxes are compared as they are
    kwargs = {"max_bbox_area_loss": None, "max_bbox_outside": None}
    expected = slc.Stream(make_transforms(), **kwargs)(slc.DataContainer((item,), fmt), return_torch=False).data[0]
    res = slc.Stream(make_transforms(), optimize_stack=True, **kwargs)(
        slc.DataContainer((item,), fmt), return_torch=False
    ).data[0]
    np.testing.assert_allclose(_as_array(res), _as_array(expected), atol=1e-4)


class _FixedProjection(slt.Projection):
    """Projection with fixed components of the perspective (vx, vy), which can have different signs."""

    serializable_name = None

    def __init__(self, vx, vy):
        super().__init__(v_range=(0, 0), p=1)
        self.v = (vx, vy)

    def sample_transform_matrix(self, data):
        super().sample_transform_matrix(data)
        self.state_dict["transform_matrix"][2, :2] = self.v


@pytest.fixture(
    params=[(1e-3, 1e-3), (-1e-3, -1e-3), (2e-3, 2e-3), (-1e-3, 1e-3), (1.5e-3, -5e-4)],
    ids=["positive", "negative", "strong", "mixed_signs", "mixed_signs_2"],
)
def projection_v(request):
    return request.param


@pytest.fixture(params=[(200, 300), (300, 200), (151, 151)], ids=["landscape", "portrait", "square_odd"])
def projection_shape(request):
    return request.param


@pytest.fixture(params=["projection_alone", "fused_after_rotation"])
def projection_pipeline(request):
    return request.param


def test_projection_frame_contains_the_warped_image(projection_v, projection_shape, projection_pipeline):
    h, w = projection_shape
    # Pixel centres of the corners, and blobs a bit inside of them
    corners = np.array([[0.0, 0.0], [w - 1, 0.0], [w - 1, h - 1], [0.0, h - 1]])
    inner = np.array([[8.0, 8.0], [w - 9, 8.0], [w - 9, h - 9], [8.0, h - 9]])
    img = np.zeros((h, w), dtype=np.float32)
    # Small blobs: the centroid of a blob moves from the warped centre with the nonlinearity of the perspective
    for x, y in inner:
        cv2.circle(img, (int(x), int(y)), 1, 1.0, -1)

    projection = _FixedProjection(*projection_v)
    if projection_pipeline == "projection_alone":
        stream = slc.Stream([projection])
    else:
        stream = slc.Stream([slt.Rotate(angle_range=(10, 10), p=1), projection], optimize_stack=True)

    kpts = slc.Keypoints(np.vstack([corners, inner]), h, w)
    dc = slc.DataContainer((img, kpts), "MP", {0: {"interpolation": "bilinear"}})
    img_res, kpts_res = stream(dc, return_torch=False).data
    h_new, w_new = img_res.shape[:2]

    warped_corners = kpts_res.data[:4]
    # The frame contains the warped image (the edges of the corner pixels are half a pixel further)
    assert np.all(warped_corners >= -1)
    assert np.all(warped_corners[:, 0] <= w_new) and np.all(warped_corners[:, 1] <= h_new)
    # The frame is tight: the image touches its left and top borders
    assert warped_corners[:, 0].min() <= 1 and warped_corners[:, 1].min() <= 1
    # The content near every corner is still in the image, where the keypoints are
    for point in kpts_res.data[4:]:
        np.testing.assert_allclose(_blob_centroid(img_res, point), point, atol=0.3)


def _blob_centroid(img, point, radius=6):
    x0, y0 = int(round(point[0])), int(round(point[1]))
    ys, xs = np.mgrid[: img.shape[0], : img.shape[1]]
    near = (np.abs(xs - x0) <= radius) & (np.abs(ys - y0) <= radius)
    weights = img * near
    assert weights.sum() > 0, f"no image content around the keypoint {point}"
    return np.array([(xs * weights).sum(), (ys * weights).sum()]) / weights.sum()
