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


@pytest.mark.parametrize("parameter", [(1, 2, 3), (10, "inherit"), (1, "i"),])
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
                slc.Stream([slt.Rotate(angle_range=(-6, 90), p=0.2, padding="r", interpolation="nearest"),])
            ),
        ]
    )

    serialized = ppl.to_dict()
    assert "interpolation" in serialized
    assert "padding" in serialized
    assert "optimize_stack" in serialized
    assert "transforms" in serialized
    assert "shuffle" in serialized
    assert len(serialized) == 6

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
        slu.from_dict({"stream": {"transforms": [{"mycrop": {"crop_to": 32, "crop_mode": "c"}},]}})

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
                slc.Stream([slt.Rotate(angle_range=(-6, 90), p=0.2, padding="r", interpolation="nearest"),])
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
