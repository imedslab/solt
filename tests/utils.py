import numpy as np
import solt.core as slc
import solt.transforms as slt


def gen_gs_img_black_edge(h, w):
    img = np.ones((h, w, 1))
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img.astype(np.uint8) * 255


def assert_data_containers_equal(dc, dc_new):
    assert dc_new.data_format == dc.data_format
    for d1, d2 in zip(dc_new.data, dc.data):
        if isinstance(d1, np.ndarray):
            np.testing.assert_array_equal(d1, d2)
        elif isinstance(d1, slc.Keypoints):
            assert d1.height == d2.height and d1.width == d2.width
            np.testing.assert_array_equal(d1.data, d2.data)
        else:
            assert d1 == d2


def other_padding(padding):
    return "r" if padding == "z" else "z"


def rotate_30(**kwargs):
    return slt.Rotate(angle_range=(30, 30), p=1, **kwargs)
