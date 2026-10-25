import random

import cv2
import numpy as np
import pytest

import solt.core as slc
import solt.transforms as slt
import solt.utils as slu

from .fixtures import *
from .utils import assert_data_containers_equal, other_padding, rotate_30


def test_fusion_happens(img_5x5):
    ppl = slc.Stream(
        [
            slt.Scale((0.5, 1.5), (0.5, 1.5), p=1),
            slt.Rotate((-50, 50), padding="z", p=1),
            slt.Shear((-0.5, 0.5), (-0.5, 0.5), padding="z", p=1),
        ]
    )
    dc = slc.DataContainer(img_5x5, "I")
    st = ppl.optimize_transforms_stack(ppl.transforms, dc)
    assert len(st) == 1


def test_stream_settings():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Shear(0.1, 0.1, interpolation="bilinear", padding="z"),
        ],
        interpolation="nearest",
        padding="z",
    )

    for trf in ppl.transforms:
        assert trf.interpolation[0] == "nearest"
        assert trf.padding[0] == "z"


def test_stream_settings_replacement():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Shear(0.1, 0.1, interpolation="bilinear", padding="z"),
        ],
        interpolation="nearest",
        padding="z",
    )

    ppl.reset_interpolation("bilinear")
    ppl.reset_padding("r")

    for trf in ppl.transforms:
        assert trf.interpolation[0] == "bilinear"
        assert trf.padding[0] == "r"


def test_stream_settings_strict():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slt.Rotate((45, 45), interpolation=("bicubic", "strict"), padding=("r", "strict"), p=1),
            slt.Shear(0.1, 0.1, interpolation="bilinear", padding="z"),
        ],
        interpolation="nearest",
        padding="z",
    )

    for idx, trf in enumerate(ppl.transforms):
        if idx == 2:
            assert trf.interpolation[0] == "bicubic"
            assert trf.padding[0] == "r"
        else:
            assert trf.interpolation[0] == "nearest"
            assert trf.padding[0] == "z"


def test_stream_nested_settings():
    ppl = slc.Stream(
        [
            slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
            slt.Rotate((45, 45), padding="r", p=1),
            slc.Stream(
                [
                    slt.Rotate((45, 45), interpolation="bicubic", padding="z", p=1),
                    slt.Rotate((45, 45), padding="r", p=1),
                ],
                interpolation="bicubic",
                padding="r",
            ),
        ],
        interpolation="nearest",
        padding="z",
    )

    trfs = ppl.transforms[:2] + ppl.transforms[-1].transforms

    for idx, trf in enumerate(trfs):
        assert trf.interpolation[0] == "nearest"
        assert trf.padding[0] == "z"


@pytest.mark.parametrize("stream_ignore_fast", [True, False])
@pytest.mark.parametrize("r1_ignore_fast", [True, False])
@pytest.mark.parametrize("r2_ignore_fast", [True, False])
def test_ignore_fast_mode_for_a_stream(stream_ignore_fast, r1_ignore_fast, r2_ignore_fast):
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(90, 90), p=1, ignore_fast_mode=r1_ignore_fast),
            slt.Rotate(angle_range=(70, 75), p=1, ignore_fast_mode=r2_ignore_fast),
        ],
        ignore_fast_mode=stream_ignore_fast,
    )

    # The setting of the stream always overrides the one of the transforms
    for trf in ppl.transforms:
        assert stream_ignore_fast == trf.ignore_fast_mode


def test_reset_ignore_fast_mode_raises_error_for_streams_for_not_bool():
    ppl = slc.Stream()
    with pytest.raises(TypeError):
        ppl.reset_ignore_fast_mode("123")


CV2_INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "lanczos": cv2.INTER_LANCZOS4,
}


CV2_PADDINGS = {"z": cv2.BORDER_CONSTANT, "r": cv2.BORDER_REFLECT_101}


# Interpolation of a transform which differs from all the stream-wide modes below
OWN_INTERPOLATION = "lanczos"


def expected_warp(img, trf, interpolation, padding):
    """Warps the image with the geometry sampled by ``trf``, using the explicitly given cv2 flags."""
    m = trf.state_dict["transform_matrix_corrected"]
    size = (trf.state_dict["w_new"], trf.state_dict["h_new"])
    res = cv2.warpAffine(img, m[:2, :], size, flags=CV2_INTERPOLATIONS[interpolation], borderMode=CV2_PADDINGS[padding])
    return res.reshape(res.shape[0], res.shape[1], -1) if img.ndim == 3 else res


def innermost(trf):
    while isinstance(trf, slc.Stream):
        trf = trf.transforms[0]
    return trf


@pytest.fixture
def mask_16x16():
    return np.random.RandomState(43).randint(0, 5, size=(16, 16)).astype(np.uint8)


@pytest.fixture(params=["z", "r"])
def stream_padding(request):
    return request.param


@pytest.fixture(params=["inherit", "strict"])
def inheritance(request):
    return request.param


@pytest.fixture(params=[0, 1, 2], ids=lambda x: f"depth{x}")
def nesting_depth(request):
    return request.param


@pytest.fixture(params=[slc.Stream, slc.SelectiveStream], ids=["Stream", "SelectiveStream"])
def nested_stream_cls(request):
    return request.param


@pytest.fixture
def wrap_in_streams(nesting_depth, nested_stream_cls):
    def wrap(trf):
        for _ in range(nesting_depth):
            trf = nested_stream_cls([trf])
        return trf

    return wrap


def test_stream_interpolation_propagates_unless_strict(
    img_16x16_rgb, stream_interpolation, inheritance, wrap_in_streams
):
    random.seed(42)
    trf = rotate_30(interpolation=(OWN_INTERPOLATION, inheritance))
    ppl = slc.Stream([wrap_in_streams(trf)], interpolation=stream_interpolation)

    expected_mode = OWN_INTERPOLATION if inheritance == "strict" else stream_interpolation
    other_mode = stream_interpolation if inheritance == "strict" else OWN_INTERPOLATION
    assert trf.interpolation == (expected_mode, inheritance)

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    np.testing.assert_array_equal(res, expected_warp(img_16x16_rgb, trf, expected_mode, "z"))
    assert not np.array_equal(res, expected_warp(img_16x16_rgb, trf, other_mode, "z"))


def test_stream_padding_propagates_unless_strict(img_16x16_rgb, stream_padding, inheritance, wrap_in_streams):
    random.seed(42)
    own_padding = other_padding(stream_padding)
    trf = rotate_30(padding=(own_padding, inheritance))
    ppl = slc.Stream([wrap_in_streams(trf)], padding=stream_padding)

    expected_padding = own_padding if inheritance == "strict" else stream_padding
    assert trf.padding == (expected_padding, inheritance)

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    np.testing.assert_array_equal(res, expected_warp(img_16x16_rgb, trf, "bilinear", expected_padding))
    assert not np.array_equal(res, expected_warp(img_16x16_rgb, trf, "bilinear", other_padding(expected_padding)))


def test_stream_settings_mix_strict_and_inherited_transforms_in_one_nested_stream(
    img_16x16_rgb, stream_interpolation, stream_padding, nested_stream_cls
):
    random.seed(42)
    strict_trf = rotate_30(
        interpolation=(OWN_INTERPOLATION, "strict"), padding=(other_padding(stream_padding), "strict")
    )
    inherit_trf = rotate_30(interpolation=OWN_INTERPOLATION, padding=other_padding(stream_padding))
    ppl = slc.Stream(
        [nested_stream_cls([strict_trf]), slc.Stream([inherit_trf])],
        interpolation=stream_interpolation,
        padding=stream_padding,
    )
    assert strict_trf.interpolation == (OWN_INTERPOLATION, "strict")
    assert strict_trf.padding == (other_padding(stream_padding), "strict")
    assert inherit_trf.interpolation == (stream_interpolation, "inherit")
    assert inherit_trf.padding == (stream_padding, "inherit")

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    expected = expected_warp(img_16x16_rgb, strict_trf, OWN_INTERPOLATION, other_padding(stream_padding))
    expected = expected_warp(expected, inherit_trf, stream_interpolation, stream_padding)
    np.testing.assert_array_equal(res, expected)


def test_stream_settings_do_not_affect_transforms_without_interpolation_and_padding(
    img_16x16_rgb, stream_interpolation, stream_padding
):
    random.seed(42)
    flip = slt.Flip(axis=1, p=1)
    brightness = slt.Brightness(brightness_range=(20, 20), p=1)
    trf = rotate_30()
    ppl = slc.Stream([flip, brightness, trf], interpolation=stream_interpolation, padding=stream_padding)

    for t in (flip, brightness):
        assert not hasattr(t, "interpolation")
        assert not hasattr(t, "padding")

    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    flipped = slc.Stream([slt.Flip(axis=1, p=1), slt.Brightness(brightness_range=(20, 20), p=1)])(
        {"image": img_16x16_rgb}, return_torch=False
    ).data[0]
    np.testing.assert_array_equal(res, expected_warp(flipped, trf, stream_interpolation, stream_padding))


def test_stream_reset_interpolation_and_padding_after_construction(img_16x16_rgb, inheritance, wrap_in_streams):
    random.seed(42)
    trf = rotate_30(interpolation=(OWN_INTERPOLATION, inheritance), padding=("z", inheritance))
    ppl = slc.Stream([wrap_in_streams(trf)])
    assert ppl.interpolation is None and ppl.padding is None
    assert trf.interpolation == (OWN_INTERPOLATION, inheritance)

    ppl.reset_interpolation("nearest")
    ppl.reset_padding("r")
    # None means "do not change"
    ppl.reset_interpolation(None)
    ppl.reset_padding(None)
    assert ppl.interpolation == "nearest" and ppl.padding == "r"

    strict = inheritance == "strict"
    expected_mode, expected_padding = (OWN_INTERPOLATION, "z") if strict else ("nearest", "r")
    res = ppl({"image": img_16x16_rgb}, return_torch=False).data[0]
    np.testing.assert_array_equal(res, expected_warp(img_16x16_rgb, trf, expected_mode, expected_padding))


def test_data_container_item_settings_win_over_stream_settings(
    img_16x16_rgb, mask_16x16, stream_interpolation, stream_padding, inheritance, nested_stream_cls
):
    random.seed(42)
    item_interpolation = "bicubic" if stream_interpolation == "nearest" else "nearest"
    item_padding = other_padding(stream_padding)
    trf = rotate_30(interpolation=(OWN_INTERPOLATION, inheritance))
    ppl = slc.Stream([nested_stream_cls([trf])], interpolation=stream_interpolation, padding=stream_padding)

    dc = slc.DataContainer(
        (img_16x16_rgb, img_16x16_rgb, mask_16x16),
        "IIM",
        transform_settings={0: {"interpolation": item_interpolation, "padding": item_padding}},
    )
    res = ppl(dc, return_torch=False)

    trf_mode = OWN_INTERPOLATION if inheritance == "strict" else stream_interpolation
    # Item 0: user-given settings are strict
    np.testing.assert_array_equal(res.data[0], expected_warp(img_16x16_rgb, trf, item_interpolation, item_padding))
    # Item 1: no item settings, the transform settings are used
    np.testing.assert_array_equal(res.data[1], expected_warp(img_16x16_rgb, trf, trf_mode, stream_padding))
    # Item 2: masks are always interpolated with nearest neighbours
    np.testing.assert_array_equal(res.data[2], expected_warp(mask_16x16, trf, "nearest", stream_padding))


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_stream_settings_survive_serialization(img_16x16_rgb, mask_16x16, stream_interpolation, stream_padding, fmt):
    def make_leaves():
        return [
            rotate_30(interpolation=(OWN_INTERPOLATION, "strict"), padding=(other_padding(stream_padding), "strict")),
            rotate_30(interpolation=OWN_INTERPOLATION, padding=other_padding(stream_padding)),
            rotate_30(),
        ]

    a, b, c = make_leaves()
    ppl = slc.Stream(
        [a, slt.Flip(axis=1, p=1), slc.SelectiveStream([slc.Stream([b])]), slc.Stream([slc.SelectiveStream([c])])],
        interpolation=stream_interpolation,
        padding=stream_padding,
    )
    if fmt == "yaml":
        ppl_new = slu.from_yaml(ppl.to_yaml())
    else:
        ppl_new = slu.from_json(ppl.to_json())

    assert ppl_new.interpolation == stream_interpolation
    assert ppl_new.padding == stream_padding
    a_new, b_new, c_new = ppl_new.transforms[0], innermost(ppl_new.transforms[2]), innermost(ppl_new.transforms[3])
    for old, new in ((a, a_new), (b, b_new), (c, c_new)):
        assert new is not old
        assert tuple(new.interpolation) == old.interpolation
        assert tuple(new.padding) == old.padding
    assert a_new.interpolation == (OWN_INTERPOLATION, "strict")
    assert b_new.interpolation == (stream_interpolation, "inherit")
    assert c_new.padding == (stream_padding, "inherit")

    data = {"image": img_16x16_rgb, "mask": mask_16x16}
    random.seed(42)
    res = ppl(data, return_torch=False)
    random.seed(42)
    res_new = ppl_new(data, return_torch=False)
    assert_data_containers_equal(res, res_new)


@pytest.mark.parametrize("ignore_fast_mode", [False, True])
def test_stream_ignore_fast_mode_propagates_to_matrix_transforms(img_16x16_rgb, ignore_fast_mode):
    random.seed(42)
    trf = rotate_30(ignore_fast_mode=not ignore_fast_mode)
    flip = slt.Flip(axis=1, p=1)
    ppl = slc.Stream([flip, trf], ignore_fast_mode=ignore_fast_mode)
    assert trf.ignore_fast_mode is ignore_fast_mode
    assert not hasattr(flip, "ignore_fast_mode")

    def out_shape():
        return ppl({"image": img_16x16_rgb}, return_torch=False).data[0].shape

    # The full geometric transform enlarges the frame to fit the rotated image, the fast mode keeps it
    assert (out_shape() != img_16x16_rgb.shape) is ignore_fast_mode

    ppl.reset_ignore_fast_mode(not ignore_fast_mode)
    assert trf.ignore_fast_mode is not ignore_fast_mode
    assert (out_shape() != img_16x16_rgb.shape) is not ignore_fast_mode


def _rotate_fast(ignore_fast_mode=False):
    return slt.Rotate(angle_range=(30, 30), p=1, ignore_fast_mode=ignore_fast_mode)


# Pipelines with the rotation nested at different depths: factories taking the outer stream kwargs
_NESTED_FAST_MODE_PIPELINES = {
    "depth_0": lambda rotate, **kw: slc.Stream([rotate], **kw),
    "depth_1": lambda rotate, **kw: slc.Stream([slc.Stream([rotate])], **kw),
    "depth_2": lambda rotate, **kw: slc.Stream([slc.Stream([slc.Stream([rotate])])], **kw),
    "selective": lambda rotate, **kw: slc.Stream([slc.SelectiveStream([rotate])], **kw),
}


@pytest.fixture(params=list(_NESTED_FAST_MODE_PIPELINES), ids=list(_NESTED_FAST_MODE_PIPELINES))
def nested_fast_mode_pipeline(request):
    return _NESTED_FAST_MODE_PIPELINES[request.param]


def _frame_enlarged(stream, img):
    return stream({"image": img}, return_torch=False).data[0].shape != img.shape


def test_stream_ignore_fast_mode_reaches_nested_streams(nested_fast_mode_pipeline):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    rotate = _rotate_fast()
    stream = nested_fast_mode_pipeline(rotate, ignore_fast_mode=True)
    assert rotate.ignore_fast_mode is True
    assert _frame_enlarged(stream, img)


def test_stream_default_overrides_the_fast_mode_of_transforms(nested_fast_mode_pipeline):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    rotate = _rotate_fast(ignore_fast_mode=True)
    stream = nested_fast_mode_pipeline(rotate)
    assert rotate.ignore_fast_mode is False
    assert stream.ignore_fast_mode is False
    assert not _frame_enlarged(stream, img)


@pytest.mark.parametrize("outer, inner", [(False, True), (True, False)])
def test_outer_stream_overrides_the_fast_mode_of_nested_streams(outer, inner):
    rotate = _rotate_fast(ignore_fast_mode=inner)
    nested = slc.Stream([rotate], ignore_fast_mode=inner)
    slc.Stream([nested], ignore_fast_mode=outer)
    assert rotate.ignore_fast_mode is outer
    assert nested.ignore_fast_mode is outer


@pytest.mark.parametrize("value", [True, False])
def test_reset_ignore_fast_mode_is_applied_recursively(nested_fast_mode_pipeline, value):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    rotate = _rotate_fast(ignore_fast_mode=not value)
    stream = nested_fast_mode_pipeline(rotate, ignore_fast_mode=not value)
    stream.reset_ignore_fast_mode(value)
    assert rotate.ignore_fast_mode is value
    assert stream.ignore_fast_mode is value
    assert _frame_enlarged(stream, img) is value
