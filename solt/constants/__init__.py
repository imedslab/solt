"""
Constants module

.. data:: ALLOWED_BLURS

    Defines the blur modes. Can be ``'g'`` - gaussian or ``'m'`` - median.

.. data:: ALLOWED_TYPES

    Defines the allowed types to be stored in a DataCantainer. Can be
    ``'I'`` - image, ``'M'`` - mask, ``'L'`` - labels, ``'P'`` - Keypoints.

.. data:: ALLOWED_CROPS

    Defines the allowed crops. Can be ``'r'`` - random crop or ``'c'`` - center crop.

.. data:: ALLOWED_PADDINGS

    Defines the allowed crops. Can be ``'z'`` - zero padding or ``'r'`` - reflective padding.

.. data:: ALLOWED_INTERPOLATIONS

    Defines the allowed interpolation modes. Can be ``'bilinear'``, ``'nearest'`` or ``'bicubic'``.

.. data:: DTYPES_MAX

    Defines the maximums for different data types. Can be ``numpy.uint8`` or ``numpy.uint16``.

.. data:: ALLOWED_COLOR_CONVERSIONS

    Defines the allowed color conversion modes. Can be ``'gs2rgb'``, ``'rgb2gs'`` or ``'none'``.


"""

from ._constants import (
    ALLOWED_BLURS,
    ALLOWED_TYPES,
    ALLOWED_CROPS,
    ALLOWED_PADDINGS,
    ALLOWED_INTERPOLATIONS_2D,
    DTYPES_MAX,
    ALLOWED_COLOR_CONVERSIONS,
    ALLOWED_GRIDMASK_MODES,
)

__all__ = [
    "ALLOWED_BLURS",
    "ALLOWED_TYPES",
    "ALLOWED_CROPS",
    "ALLOWED_PADDINGS",
    "ALLOWED_INTERPOLATIONS_2D",
    "DTYPES_MAX",
    "ALLOWED_COLOR_CONVERSIONS",
    "ALLOWED_GRIDMASK_MODES",
]
