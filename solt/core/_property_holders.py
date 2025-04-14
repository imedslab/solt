from solt.utils import validate_parameter
from solt.constants import ALLOWED_PADDINGS, ALLOWED_INTERPOLATIONS_2D


class PaddingPropertyHolder(object):
    """
    PaddingPropertyHolder

    Adds padding property to a class and validates it using the allowed paddings from constants.

    Parameters
    ----------
    padding : None or str or tuple
        Padding mode. Inheritance can be specified as the second argument of the `padding` tuple.

    """

    def __init__(self, padding=None):
        super(PaddingPropertyHolder, self).__init__()
        self.padding = validate_parameter(padding, ALLOWED_PADDINGS, "z")


class InterpolationPropertyHolder(object):
    """
    InterpolationPropertyHolder

    Adds interpolation property to a class and validates it using the allowed interpolations from constants.

    Parameters
    ----------
    interpolation : None or str or tuple
        Interpolation mode. Inheritance can be specified as the second argument of the `interpolation` tuple.

    """

    def __init__(self, interpolation=None):
        super(InterpolationPropertyHolder, self).__init__()
        self.interpolation = validate_parameter(interpolation, ALLOWED_INTERPOLATIONS_2D, "bilinear")
