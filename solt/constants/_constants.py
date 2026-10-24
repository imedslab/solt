import cv2
import numpy as np

ALLOWED_PADDINGS = {"z": cv2.BORDER_CONSTANT, "r": cv2.BORDER_REFLECT_101}
ALLOWED_INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}
ALLOWED_CROPS = {"c", "r", "lt", "rt", "lb", "rb"}
ALLOWED_TYPES = {"I", "M", "D", "P", "B", "L"}
ALLOWED_BLURS = {"g", "m", "mo"}
ALLOWED_COLOR_CONVERSIONS = {"gs2rgb", "rgb2gs", "none"}
DTYPES_MAX = {np.dtype(dtype): np.iinfo(dtype).max for dtype in (np.uint8, np.uint16)}
ALLOWED_GRIDMASK_MODES = {"crop", "reserve", "none"}
