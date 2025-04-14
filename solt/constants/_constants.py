import cv2
import numpy as np

ALLOWED_PADDINGS_2D = {"z": cv2.BORDER_CONSTANT, "r": cv2.BORDER_REFLECT_101}
ALLOWED_INTERPOLATIONS_2D = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

ALLOWED_PADDINGS_3D = {"z": 0, "r": 1}

ALLOWED_INTERPOLATIONS_3D = {
    "trilinear": 0,
    "tricubic": 1,
    "nearest": 2,
}

ALLOWED_INTERPOLATIONS_2D = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}
ALLOWED_CROPS = {"c", "r"}
ALLOWED_TYPES = {"I", "M", "P", "L", "V", "VM"}
# V: Volume
# M: Mask
# VM: Volumetric Mask
ALLOWED_BLURS = {"g", "m", "mo"}
ALLOWED_COLOR_CONVERSIONS = {"gs2rgb", "rgb2gs", "none"}
DTYPES_MAX = {np.dtype("uint8"): 255, np.dtype("uint16"): 65536}
ALLOWED_GRIDMASK_MODES = {"crop", "reserve", "none"}
