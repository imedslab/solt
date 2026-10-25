import pytest
import numpy as np
from .utils import gen_gs_img_black_edge

__all__ = [
    "img_2x2",
    "mask_2x2",
    "img_3x3_rgb",
    "img_3x3",
    "img_3x4",
    "img_3x4_rgb",
    "img_5x5",
    "img_6x6",
    "img_6x6_lc",
    "img_6x6_rgb",
    "img_7x7",
    "mask_3x3",
    "mask_3x4",
    "mask_5x5",
    "mask_6x6",
    "img_16x16_rgb",
    "stream_interpolation",
]


@pytest.fixture
def img_2x2():
    """
    Generates a 2x2 grayscale image (uint8)

    Returns
    -------
    out : ndarray
        2x2x1 uint8 image
    """
    return np.array([[1, 0], [1, 1]]).reshape((2, 2, 1)).astype(np.uint8)


@pytest.fixture
def mask_2x2():
    """
    Generates 2x2 mask (doesn't have the 3rd dimension compare to an image).

    Returns
    -------
    out : ndarray
        2x2 mask, uint8
    """
    return np.array([[1, 0], [0, 1]]).reshape((2, 2)).astype(np.uint8)


@pytest.fixture
def img_3x4():
    """
    Generates a grayscale image 3x4

    Returns
    -------
    out : ndarray
        3x4x1 uint8 image
    """
    img = np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1]]).reshape((3, 4, 1)).astype(np.uint8) * 255
    return img


@pytest.fixture
def img_3x4_rgb():
    """
    Generates a grayscale image 3x4

    Returns
    -------
    out : ndarray
        3x4x1 uint8 image
    """
    img = np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1]]).reshape((3, 4, 1)).astype(np.uint8) * 255
    return np.dstack((img, img, img))


@pytest.fixture
def mask_3x4():
    """
    Generates a mask  3x4

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """

    mask = np.array([[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0]]).reshape((3, 4)).astype(np.uint8)
    return mask


@pytest.fixture
def img_3x3():
    """
    Generates a grayscale image 3x4

    Returns
    -------
    out : ndarray
        3x4x1 uint8 image
    """
    img = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]]).reshape((3, 3, 1)).astype(np.uint8)
    return img


@pytest.fixture
def img_3x3_rgb():
    """
    Generates a grayscale image 3x4

    Returns
    -------
    out : ndarray
        3x4x1 uint8 image
    """
    img = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]]).reshape((3, 3, 1)).astype(np.uint8)
    return np.dstack((img, img, img)) * 255


@pytest.fixture
def mask_3x3():
    """
    Generates a image+mask  3x4

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """

    mask = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1]]).reshape((3, 3)).astype(np.uint8)
    return mask


@pytest.fixture
def img_5x5():
    """
    Generates a gs image 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        5x5 uint8 image
    """
    return gen_gs_img_black_edge(5, 5) // 255


@pytest.fixture
def mask_5x5():
    """
    Generates a mask 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        5x5 uint8 image
    """
    img = np.ones((5, 5, 1))

    img[:, :2] = 2
    img[:, -2:] = 2
    img[:2, :] = 2
    img[-2, :] = 2
    return img.astype(np.uint8)


@pytest.fixture
def img_6x6():
    """
    Generates a gs image 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        6x6 uint8 image
    """
    return gen_gs_img_black_edge(6, 6)


@pytest.fixture
def img_7x7():
    """
    Generates a gs image 7x7. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        6x6 uint8 image
    """

    return gen_gs_img_black_edge(7, 7)


@pytest.fixture
def mask_6x6():
    """
    Generates a mask 6x6. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        3x5 uint8 image
    """
    img = np.ones((6, 6))

    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img.astype(np.uint8)


@pytest.fixture
def img_6x6_rgb():
    """
    Generates an RGB image 6x6. It is all 255, besides the edges

    Returns
    -------
    out : ndarray
        6x6 uint8 image
    """
    img = gen_gs_img_black_edge(6, 6)
    return np.dstack((img, img, img))


@pytest.fixture
def img_6x6_lc():
    """
    Generates an RGB image 6x6. It is all 7x7, besides the edges (low contrast)

    Returns
    -------
    out : ndarray
        6x6 uint8 image
    """
    img = np.ones((6, 6, 1))
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return np.dstack((img, img, img)).astype(np.uint8) * 127


@pytest.fixture
def img_16x16_rgb():
    return np.random.RandomState(42).randint(0, 256, size=(16, 16, 3)).astype(np.uint8)


@pytest.fixture(params=["nearest", "bilinear", "bicubic"])
def stream_interpolation(request):
    return request.param
