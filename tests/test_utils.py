import copy
import math

import numpy as np
import solt.utils as slu
import solt.transforms as slt
import solt.core as slc
import pytest
import json
import pathlib
import yaml


def test_parameter_validation_range_default_value_not_tuple():
    with pytest.raises(TypeError):
        slu.validate_numeric_range_parameter(123, 123)


def test_parameter_validation_raises_error_when_types_dont_match():
    with pytest.raises(NotImplementedError):
        slu.validate_parameter({1, 2}, 10, int)


def test_parameter_validation_raises_error_when_default_type_is_wrong():
    with pytest.raises(ValueError):
        slu.validate_parameter(None, {1, 2}, (10, "12345"), int)


def test_parameter_validation_raises_error_when_default_value_is_wrong_type():
    with pytest.raises(TypeError):
        slu.validate_parameter(None, {1, 2}, ("10", "inherit"), int)


@pytest.mark.parametrize(
    "parameter",
    [
        (1, 2, 3),
        (10, "inherit"),
        (1, "i"),
    ],
)
def test_validate_parameter_raises_value_errors(parameter):
    with pytest.raises(ValueError):
        slu.validate_parameter(parameter, {1, 2}, 1, basic_type=int)


@pytest.mark.parametrize(
    "serialized",
    [
        {
            "stream": {
                "transforms": [
                    {"pad": {"pad_to": 34}},
                    {"crop": {"crop_to": 32, "crop_mode": "r"}},
                    {"cutout": {"cutout_size": 2}},
                ]
            }
        },
        {
            "stream": {
                "interpolation": None,
                "padding": None,
                "transforms": [
                    {"pad": {"pad_to": 34, "padding": "z"}},
                    {"crop": {"crop_to": 32, "crop_mode": "r"}},
                    {"cutout": {"cutout_size": 2, "p": 0.5}},
                ],
            },
        },
    ],
)
def test_deserialize_from_dict(serialized):
    trfs = slc.Stream([slt.Pad(34), slt.Crop(32, "r"), slt.CutOut(2)])

    serialized_trfs = json.dumps(trfs.to_dict())
    serialized_from_deserialized = json.dumps(slu.from_dict(serialized).to_dict())

    assert serialized_trfs == serialized_from_deserialized


@pytest.mark.parametrize(
    "serialized, stream",
    [
        [
            {
                "stream": {
                    "transforms": [
                        {"pad": {"pad_to": 34}},
                        {"crop": {"crop_to": 32, "crop_mode": "r"}},
                        {"cutout": {"cutout_size": 2}},
                        {
                            "stream": {
                                "interpolation": None,
                                "padding": None,
                                "transforms": [
                                    {"pad": {"pad_to": 34, "padding": "z"}},
                                    {"crop": {"crop_to": 32, "crop_mode": "r"}},
                                    {"cutout": {"cutout_size": 2, "p": 0.5}},
                                ],
                            },
                        },
                    ]
                }
            },
            slc.Stream(
                [
                    slt.Pad(34),
                    slt.Crop(32, "r"),
                    slt.CutOut(2),
                    slc.Stream([slt.Pad(34), slt.Crop(32, "r"), slt.CutOut(2)]),
                ]
            ),
        ],
        [
            {
                "stream": {
                    "transforms": [
                        {
                            "stream": {
                                "interpolation": None,
                                "padding": None,
                                "transforms": [
                                    {"pad": {"pad_to": 34, "padding": "z"}},
                                    {"crop": {"crop_to": 32, "crop_mode": "r"}},
                                    {"cutout": {"cutout_size": 2, "p": 0.5}},
                                ],
                            },
                        },
                        {"pad": {"pad_to": 34}},
                        {"crop": {"crop_to": 32, "crop_mode": "c"}},
                        {"cutout": {"cutout_size": 4}},
                        {
                            "projection": {
                                "v_range": (0, 1e-3),
                                "affine_transforms": {
                                    "stream": {
                                        "transforms": [
                                            {"rotate": {"angle_range": 30}},
                                            {"scale": {"range_x": 2, "same": True}},
                                        ],
                                    }
                                },
                            }
                        },
                    ]
                }
            },
            slc.Stream(
                [
                    slc.Stream([slt.Pad(34), slt.Crop(32, "r"), slt.CutOut(2)]),
                    slt.Pad(34),
                    slt.Crop(32, "c"),
                    slt.CutOut(4),
                    slt.Projection(slc.Stream([slt.Rotate(30), slt.Scale(2)]), v_range=(0, 1e-3)),
                ]
            ),
        ],
    ],
)
def test_deserialize_from_dict_nested(serialized: dict, stream: slc.Stream):
    serialized_trfs = json.dumps(stream.to_dict())
    serialized_from_deserialized = json.dumps(slu.from_dict(serialized).to_dict())

    assert serialized_trfs == serialized_from_deserialized


def test_stream_serializes_all_args_are_set():
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(-106, 90), p=0.7, interpolation="nearest"),
            slt.Rotate(angle_range=(-106, 90), p=0.7, interpolation="nearest"),
            slt.Rotate(angle_range=(-106, 90), p=0.7, interpolation="nearest"),
            slt.Projection(
                slc.Stream(
                    [
                        slt.Rotate(angle_range=(-6, 90), p=0.2, padding="r", interpolation="nearest"),
                    ]
                )
            ),
        ]
    )

    serialized = ppl.to_dict()
    assert "interpolation" in serialized
    assert "padding" in serialized
    assert "optimize_stack" in serialized
    assert "transforms" in serialized
    assert "shuffle" in serialized
    assert "in_place" in serialized
    assert "max_bbox_area_loss" in serialized
    assert "max_bbox_outside" in serialized
    assert len(serialized) == 9

    trfs = serialized["transforms"]
    for i, el in enumerate(trfs):
        t = list(el.keys())[0]
        if i < len(trfs) - 1:
            assert list(el.keys())[0] == "rotate"
            assert trfs[i][t]["p"] == 0.7
            assert trfs[i][t]["interpolation"] == ("nearest", "inherit")
            assert trfs[i][t]["padding"] == ("z", "inherit")
            assert trfs[i][t]["angle_range"] == (-106, 90)
        else:
            assert t == "projection"
            assert trfs[i][t]["affine_transforms"]["stream"]["transforms"][0]["rotate"]["p"] == 0.2
            assert trfs[i][t]["affine_transforms"]["stream"]["transforms"][0]["rotate"]["interpolation"] == (
                "nearest",
                "inherit",
            )
            assert trfs[i][t]["affine_transforms"]["stream"]["transforms"][0]["rotate"]["padding"] == ("r", "inherit")
            assert trfs[i][t]["affine_transforms"]["stream"]["transforms"][0]["rotate"]["angle_range"] == (-6, 90)


def test_transforms_type_when_deserializing():
    with pytest.raises(TypeError):
        slu.from_dict(2)


def test_transforms_not_in_registry_and_in_when_initialized():
    with pytest.raises(ValueError):
        slu.from_dict(
            {
                "stream": {
                    "transforms": [
                        {"mycrop": {"crop_to": 32, "crop_mode": "c"}},
                    ]
                }
            }
        )

    assert "mycrop" not in slc.BaseTransform.registry

    class MyCrop(slc.BaseTransform):
        serializable_name = "mycrop"

    assert "mycrop" in slc.BaseTransform.registry


def test_to_from_yaml_json():
    ppl = slc.Stream(
        [
            slt.Rotate(angle_range=(-106, 90), p=0.7, interpolation="nearest"),
            slt.Rotate(angle_range=(-106, 91), p=0.8, interpolation="nearest"),
            slt.Rotate(angle_range=(-106, 92), p=0.2, interpolation="bilinear"),
            slt.Projection(
                slc.Stream(
                    [
                        slt.Rotate(angle_range=(-6, 90), p=0.2, padding="r", interpolation="nearest"),
                    ]
                )
            ),
        ]
    )

    serialized = {"stream": ppl.to_dict()}
    with open("/tmp/solt_tmp_json0.json", "w") as f:
        json_s = json.dumps(serialized, indent=4)
        f.write(json_s)

    with open("/tmp/solt_tmp_yaml0.yaml", "w") as f:
        yaml_s = yaml.safe_dump(serialized)
        f.write(yaml_s)

    ppl.to_json("/tmp/solt_tmp_json1.json")
    ppl.to_yaml("/tmp/solt_tmp_yaml1.yaml")

    loaded_from_json = slu.from_json("/tmp/solt_tmp_json1.json")
    loaded_from_yaml = slu.from_yaml("/tmp/solt_tmp_yaml1.yaml")
    assert json_s == loaded_from_json.to_json()
    assert yaml_s == loaded_from_yaml.to_yaml()

    loaded_from_json = slu.from_json(pathlib.Path("/tmp/solt_tmp_json1.json"))
    loaded_from_yaml = slu.from_yaml(pathlib.Path("/tmp/solt_tmp_yaml1.yaml"))
    assert json_s == slu.from_yaml(loaded_from_json.to_yaml()).to_json()
    assert yaml_s == slu.from_json(loaded_from_yaml.to_json()).to_yaml()


def test_complex_transform_serialization():
    stream = slc.Stream(
        [
            slt.Flip(axis=1, p=0.5),
            slc.SelectiveStream(
                [
                    slt.Rotate(angle_range=(-45, -45), p=1, padding="r"),
                    slt.Rotate90(1, p=1),
                    slt.Rotate(angle_range=(45, 45), p=1, padding="r"),
                ]
            ),
            slt.Crop((350, 350)),
            slc.SelectiveStream(
                [slt.GammaCorrection(gamma_range=0.5, p=1), slt.Noise(gain_range=0.1, p=1), slt.Blur()], n=3
            ),
            slt.Projection(
                affine_transforms=slc.Stream(
                    [
                        slt.Rotate(angle_range=(-45, 45), p=1),
                        slt.Scale(range_x=(0.8, 1.5), range_y=(0.8, 1.5), p=1, same=False),
                    ]
                ),
                v_range=(1e-4, 1e-3),
                p=1,
            ),
            slc.SelectiveStream(
                [
                    slt.CutOut(40, p=1),
                    slt.CutOut(30, p=1),
                    slt.CutOut(20, p=1),
                    slt.CutOut(40, p=1),
                    slc.Stream(),
                    slc.Stream(),
                    slc.Stream(),
                ],
                n=3,
            ),
        ]
    )

    assert slu.from_yaml(stream.to_yaml()).to_yaml() == slu.from_yaml(stream.to_yaml()).to_yaml()


def brute_force_sdf(mask):
    """Signed distance field by definition: the distance to the nearest pixel of the other kind."""
    mask = mask.astype(bool)
    coords = np.argwhere(np.ones_like(mask))
    inside, outside = np.argwhere(mask), np.argwhere(~mask)
    res = np.zeros(mask.shape)
    for y, x in coords:
        other = outside if mask[y, x] else inside
        d = np.sqrt(((other - (y, x)) ** 2).sum(axis=1)).min()
        res[y, x] = -d if mask[y, x] else d
    return res


@pytest.mark.parametrize("seed", range(5))
def test_mask_to_sdf_matches_definition(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((12, 15)) > 0.7
    np.testing.assert_allclose(slu.mask_to_sdf(mask, dtype=np.float64), brute_force_sdf(mask))


def test_mask_to_sdf_sign_and_boundary():
    mask = np.zeros((1, 6), dtype=np.uint8)
    mask[0, 2:4] = 255
    np.testing.assert_array_equal(slu.mask_to_sdf(mask)[0], [2, 1, -1, -1, 1, 2])


def test_mask_to_sdf_disc_is_close_to_analytic():
    yy, xx = np.mgrid[:65, :65]
    distance_to_center = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
    sdf = slu.mask_to_sdf(distance_to_center <= 15)
    # The mask has pixel steps, so the error is up to a pixel at the boundary
    error = np.abs(sdf - (distance_to_center - 15))
    assert error.max() <= 1 + 1e-6
    assert error.mean() < 0.25


def test_mask_to_sdf_multichannel():
    rng = np.random.default_rng(0)
    masks = rng.random((10, 12, 3)) > 0.6
    sdf = slu.mask_to_sdf(masks)
    assert sdf.shape == masks.shape
    for c in range(3):
        np.testing.assert_array_equal(sdf[..., c], slu.mask_to_sdf(masks[..., c]))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(10, 12), (10, 12, 1), (10, 12, 5)])
def test_mask_to_sdf_dtype_and_shape(dtype, shape):
    mask = np.random.default_rng(0).random(shape) > 0.5
    sdf = slu.mask_to_sdf(mask, dtype=dtype)
    assert sdf.dtype == dtype
    assert sdf.shape == shape


@pytest.mark.parametrize("max_distance, expected_sign", [(None, 1), (5, 1)])
def test_mask_to_sdf_empty_mask(max_distance, expected_sign):
    sdf = slu.mask_to_sdf(np.zeros((10, 12), dtype=bool), max_distance=max_distance)
    far = math.hypot(10, 12) if max_distance is None else max_distance
    np.testing.assert_allclose(sdf, expected_sign * far, rtol=1e-6)


@pytest.mark.parametrize("max_distance", [None, 5])
def test_mask_to_sdf_full_mask(max_distance):
    sdf = slu.mask_to_sdf(np.ones((10, 12), dtype=bool), max_distance=max_distance)
    far = math.hypot(10, 12) if max_distance is None else max_distance
    np.testing.assert_allclose(sdf, -far, rtol=1e-6)


def test_mask_to_sdf_channels_without_boundary():
    masks = np.zeros((10, 12, 3), dtype=bool)
    masks[2:5, 3:7, 1] = True
    masks[..., 2] = True
    sdf = slu.mask_to_sdf(masks)
    assert np.all(sdf[..., 0] > 0)
    np.testing.assert_array_equal(sdf[..., 1], slu.mask_to_sdf(masks[..., 1]))
    assert np.all(sdf[..., 2] < 0)


def test_mask_to_sdf_max_distance():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    sdf = slu.mask_to_sdf(mask, max_distance=3)
    assert sdf.min() == -3 and sdf.max() == 3
    unclipped = slu.mask_to_sdf(mask)
    small = np.abs(unclipped) <= 3
    np.testing.assert_array_equal(sdf[small], unclipped[small])


@pytest.mark.parametrize(
    "mask, kwargs, error",
    [
        ([[0, 1], [1, 0]], {}, TypeError),
        (np.zeros(5), {}, ValueError),
        (np.zeros((2, 2, 2, 2)), {}, ValueError),
        (np.zeros((5, 5)), {"dtype": np.int32}, TypeError),
        (np.zeros((5, 5)), {"max_distance": 0}, ValueError),
        (np.zeros((5, 5)), {"max_distance": -1}, ValueError),
        (np.zeros((5, 5)), {"max_distance": "5"}, TypeError),
    ],
)
def test_mask_to_sdf_wrong_input(mask, kwargs, error):
    with pytest.raises(error):
        slu.mask_to_sdf(mask, **kwargs)


@pytest.fixture(
    params=[
        ("g", {"g", "m"}, "g"),
        (None, {"g", "m"}, "m"),
        ("m", {"g", "m"}, "m"),
    ],
    ids=["value", "default", "other_value"],
)
def valid_non_heritable(request):
    return request.param


def test_validate_non_heritable_parameter(valid_non_heritable):
    value, allowed, expected = valid_non_heritable
    default = expected if value is None else "g"
    assert slu.validate_parameter(value, allowed, default, str, heritable=False) == expected


@pytest.mark.parametrize(
    "value, error", [("xyz", ValueError), ("", ValueError), (1, TypeError), (("g", "strict"), TypeError)]
)
def test_validate_non_heritable_parameter_rejects_wrong_values(value, error):
    with pytest.raises(error):
        slu.validate_parameter(value, {"g", "m"}, "g", str, heritable=False)


@pytest.fixture(
    params=[
        (lambda mode: slt.Blur(blur_type=mode, p=1), "blur"),
        (lambda mode: slt.CvtColor(mode=mode), "cvtcolor"),
        (lambda mode: slt.GridMask(mode=mode, p=1), "gridmask"),
    ],
    ids=["Blur", "CvtColor", "GridMask"],
)
def transform_with_mode(request):
    return request.param[0]


@pytest.mark.parametrize("mode", ["xyz", "G", "rgb2bgr", 1, ("g", "strict")])
def test_transforms_reject_unknown_modes_on_creation(transform_with_mode, mode):
    with pytest.raises((ValueError, TypeError)):
        transform_with_mode(mode)


_NESTED_CONFIGS = {
    "flat": {"stream": {"transforms": [{"flip": {"axis": 1}}, {"rotate": {"angle_range": [-10, 10]}}]}},
    "nested_stream": {"stream": {"transforms": [{"stream": {"transforms": [{"flip": {"axis": 0}}]}}]}},
    "projection": {
        "stream": {
            "transforms": [
                {
                    "projection": {
                        "affine_transforms": {"stream": {"transforms": [{"rotate": {"angle_range": [-10, 10]}}]}}
                    }
                }
            ]
        }
    },
    "selective": {"selective_stream": {"n": 1, "transforms": [{"flip": {"axis": 1}}, {"flip": {"axis": 0}}]}},
    "without_transforms": {"stream": {"interpolation": "nearest"}},
    "empty_selective": {"selective_stream": {"n": 0}},
}


@pytest.fixture(params=list(_NESTED_CONFIGS), ids=list(_NESTED_CONFIGS))
def nested_config(request):
    return _NESTED_CONFIGS[request.param]


def test_from_dict_does_not_modify_its_input(nested_config):
    original = copy.deepcopy(nested_config)
    first = slu.from_dict(nested_config)
    assert nested_config == original
    # The same config can be loaded again
    assert slu.from_dict(nested_config).to_dict() == first.to_dict()


def test_from_dict_without_transforms():
    stream = slu.from_dict({"stream": {"interpolation": "nearest"}})
    assert isinstance(stream, slc.Stream)
    assert stream.transforms == []
    assert stream.interpolation == "nearest"


def test_selective_stream_without_transforms_keeps_the_data():
    img = np.full((4, 4, 1), 7, dtype=np.uint8)
    stream = slu.from_dict({"selective_stream": {"n": 0}})
    assert isinstance(stream, slc.SelectiveStream)
    res = stream({"image": img}, return_torch=False).data[0]
    np.testing.assert_array_equal(res, img)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"flip": {"axis": 1}, "rotate": {"angle_range": [1, 2]}},
        {"stream": {"transforms": [{}]}},
        {"stream": {"transforms": [{"flip": {"axis": 1}, "rotate": {"angle_range": [1, 2]}}]}},
    ],
    ids=["empty", "two_transforms", "empty_item", "item_with_two_transforms"],
)
def test_from_dict_requires_exactly_one_transform_per_dict(config):
    with pytest.raises(ValueError):
        slu.from_dict(config)
