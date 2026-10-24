import numpy as np

from ._base_transforms import (
    BaseTransform,
    MatrixTransform,
)
import copy
import random

from solt.utils import Serializable
import numbers

from ._data import BoundingBoxes, DataContainer


class Stream(Serializable):
    """
    Stream class. Executes the list of transformations.

    The stream can be called directly on a dict of items,
    which get converted into a DataContainer. Note that the DataContainer assumes
    **the same** coordinate system for all the items it is storing.
    For instance, the behavior shown below is not supported for any stream:

    .. highlight:: python
    .. code-block:: python

        import solt.transforms as slt
        import numpy as np

        trf = solt.Stream([slt.Resize(resize_to=(50, 50)),
                               slt.Flip()])
        img1 = np.ones((100, 100, 3), dtype=np.int32)
        img2 = np.ones((110, 110, 3), dtype=np.int32)
        trf({'images': (img1, img2)})

    Parameters
    ----------
    transforms : list or None
        List of transforms to execute
    interpolation : str or None
        Stream-wide settings for interpolation. If for some particular transform your would like
        to still use its own mode, simply pass ``(<interpolation_value>, 'strict')``
        in the constructor of that transform.
    padding : str or None
        Stream-wide settings for padding. If for some particular transform your would like
        to still use its own mode, simply pass ``(<padding_value>, 'strict')``
        in the constructor of that transform.
    optimize_stack : bool
        Whether to run transforms stack optimization. It can only be useful if many matrix transformations are
        in a row.
    ignore_fast_mode : bool
        Whether to ignore the fast mode. This option enables full geometric transforms.
    shuffle : bool
        Whether to shuffle the transforms before applying them.
    in_place : bool
        Whether the transforms may modify the input arrays in place. Transforms do not copy the data unless
        necessary, e.g. ``CutOut`` writes into the image it receives, and ``Crop`` returns a view. If ``False``, the
        stream copies the input data once before applying the transforms, so the original arrays stay unchanged.
    max_bbox_area_loss : float or None
        Bounding boxes are dropped (with their labels), if their share of the image area at the end of the stream is
        smaller than at its start by more than this fraction. ``0.9`` by default, i.e. the boxes, which became more
        than 10 times smaller relative to the image, are dropped. Resizing the whole image does not change the share,
        while e.g. ``Scale`` in the same frame or ``Pad`` decrease it. ``None`` disables this check.
    max_bbox_outside : float or None
        Bounding boxes are dropped (with their labels), if more than this fraction of their area is outside the image
        at the end of the stream. ``0.5`` by default. ``None`` disables this check.

    Notes
    -----
    ``DataContainer`` clips the bounding boxes to the image after every transform, and tracks the unclipped ones.
    At the end, the stream drops the boxes, which exceed ``max_bbox_area_loss`` or ``max_bbox_outside``, and the boxes
    fully outside of the image, also if both checks are disabled. Nested streams leave this to the outermost one.

    """

    serializable_name = "stream"
    """
    How the class should be stored in the registry"""

    def __init__(
        self,
        transforms=None,
        interpolation=None,
        padding=None,
        optimize_stack=False,
        ignore_fast_mode=False,
        shuffle=False,
        in_place=True,
        max_bbox_area_loss=0.9,
        max_bbox_outside=0.5,
    ):
        super(Stream, self).__init__()

        if transforms is None:
            transforms = []

        for trf in transforms:
            if not isinstance(trf, BaseTransform) and not isinstance(trf, Stream):
                raise TypeError("Unknown transform type found in the Stream!")
        self.shuffle = shuffle
        self.in_place = in_place
        self.max_bbox_area_loss = Stream._validate_fraction(max_bbox_area_loss, "max_bbox_area_loss")
        self.max_bbox_outside = Stream._validate_fraction(max_bbox_outside, "max_bbox_outside")
        self.optimize_stack = optimize_stack
        self.interpolation = interpolation
        self.padding = padding
        self.ignore_fast_mode = ignore_fast_mode
        self.transforms = transforms

        self.reset_ignore_fast_mode(ignore_fast_mode)
        self.reset_padding(padding)
        self.reset_interpolation(interpolation)

    def reset_ignore_fast_mode(self, value):
        if not isinstance(value, bool):
            raise TypeError("Ignore fast mode must be bool!")
        for trf in self.transforms:
            if isinstance(trf, MatrixTransform):
                trf.ignore_fast_mode = value

    def reset_interpolation(self, value):
        """
        Resets the interpolation for the whole pipeline of transforms.

        Parameters
        ----------
        value : str or None
            A value from ``solt.constants.ALLOWED_INTERPOLATIONS``

        See also
        --------
        solt.constants.ALLOWED_INTERPOLATIONS

        """
        if value is None:
            return
        self.interpolation = value
        for trf in self.transforms:
            if self.interpolation is not None and hasattr(trf, "interpolation"):
                if isinstance(trf, BaseTransform):
                    if trf.interpolation[1] != "strict":
                        trf.interpolation = (self.interpolation, trf.interpolation[1])
                elif isinstance(trf, Stream):
                    trf.reset_interpolation(self.interpolation)

    def reset_padding(self, value):
        """
        Allows to reset the padding for the whole Stream

        Parameters
        ----------
        value : str
            Should be a string from ``solt.constants.ALLOWED_PADDINGS``

        See also
        --------
        solt.constants.ALLOWED_PADDINGS

        """
        if value is None:
            return
        self.padding = value
        for trf in self.transforms:
            if self.padding is not None and hasattr(trf, "padding"):
                if isinstance(trf, BaseTransform):
                    if trf.padding[1] != "strict":
                        trf.padding = (self.padding, trf.padding[1])
                elif isinstance(trf, Stream):
                    trf.reset_padding(self.padding)

    def __call__(
        self,
        data,
        return_torch=True,
        as_dict=True,
        scale_keypoints=True,
        normalize=True,
        mean=None,
        std=None,
    ):
        """
        Executes the list of the pre-defined transformations for a given data container.

        Parameters
        ----------
        data : DataContainer or dict
            Data to be augmented. See ``solt.core.DataContainer.from_dict`` for details.
        return_torch : bool
            Whether to convert the result into a torch tensors. By default, it is false for transforms and
            true for the streams.
        as_dict : bool
            Whether to pool the results into a dict. See ``solt.core.DataContainer.to_dict`` for details
        scale_keypoints : bool
            Whether to scale the keypoints into 0-1 range
        normalize : bool
            Whether to normalize the resulting tensor. If mean or std args are None,
            ImageNet statistics will be used
        mean : None or tuple of float or np.ndarray or torch.FloatTensor
            Mean to subtract for the converted tensor
        std : None or tuple of float or np.ndarray or torch.FloatTensor
            Mean to subtract for the converted tensor

        Returns
        -------
        out : DataContainer or dict or list
            Result

        """

        data = self.wrap_data(data)
        bboxes = [item for item, t, _ in data if t == "B"]
        res: DataContainer = self._apply(data)
        if bboxes:
            res = self.filter_bboxes(bboxes, res)

        if return_torch:
            return res.to_torch(
                as_dict=as_dict,
                scale_keypoints=scale_keypoints,
                normalize=normalize,
                mean=mean,
                std=std,
            )
        return res

    @staticmethod
    def optimize_transforms_stack(transforms, data):
        """
        Static method which fuses the transformations

        Parameters
        ----------
        transforms : list
            A list of transforms
        data : DataContainer
            Data container to be used to sample the transforms

        Returns
        -------
        out : list
            An optimized list of transforms

        """
        # First we should create a stack
        transforms_stack = []
        for trf in transforms:
            if isinstance(trf, MatrixTransform):
                # The transforms of the stream must not change: the first one would get the matrices, the interpolation
                # and the padding of the others. A shallow copy with its own state is enough, no arrays are copied.
                trf = copy.copy(trf)
                trf.ignore_fast_mode = True
                trf.reset_state()
                if trf.use_transform():
                    trf.sample_transform(data)
                    if len(transforms_stack) == 0:
                        transforms_stack.append(trf)
                    else:
                        transforms_stack[-1].fuse_with(trf)
            else:
                raise TypeError("Nested streams or other transforms but the `Matrix` ones are not supported!")

        if len(transforms_stack) > 0:
            transforms_stack[-1].correct_transform()
        return transforms_stack

    @staticmethod
    def _validate_fraction(value, name):
        if value is None:
            return None
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number or None!")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1!")
        return value

    def _apply(self, data):
        """
        Applies the transforms of the stream to a data container, without the final processing of ``__call__``.

        Parameters
        ----------
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result

        """
        return Stream.exec_stream(self.transforms, data, self.optimize_stack, self.shuffle)

    def filter_bboxes(self, bboxes_before, data):
        """
        Clips the bounding boxes to the image, and drops the ones, which exceed ``max_bbox_area_loss`` or
        ``max_bbox_outside``, together with their labels.

        Parameters
        ----------
        bboxes_before : list of BoundingBoxes
            Bounding boxes of the data container before the transforms, in the same order.
        data : DataContainer
            Data container after the transforms.

        Returns
        -------
        out : DataContainer
            Data container with the clipped and filtered bounding boxes.

        """
        items = []
        bboxes_before = iter(bboxes_before)
        for item, t, _ in data:
            if t == "B":
                item = self._filter_bboxes(next(bboxes_before), item)
            items.append(item)
        return DataContainer(tuple(items), data.data_format, data.transform_settings)

    def _filter_bboxes(self, before, after):
        # DataContainer clips the boxes after every transform, the unclipped ones show, how much is outside
        clipped, unclipped = after.data, after.unclipped
        areas, clipped_areas = after.unclipped_areas(), after.areas()

        # Boxes, whose content has left the image, are always dropped. A box without area is outside, if it was moved
        lost = (clipped_areas == 0) & ((areas > 0) | np.any(clipped != unclipped, axis=1))
        keep = ~lost
        if self.max_bbox_area_loss is not None:
            # Shares of the image area, so that resizing the whole image does not change them
            share_before = before.areas() / (before.height * before.width)
            share_after = clipped_areas / (after.height * after.width)
            keep &= share_after >= (1 - self.max_bbox_area_loss) * share_before
        if self.max_bbox_outside is not None:
            # Boxes without an area are outside if clipping moves them
            outside = np.where(
                areas > 0,
                1 - clipped_areas / np.where(areas > 0, areas, 1),
                np.any(clipped != unclipped, axis=1).astype(float),
            )
            keep &= outside <= self.max_bbox_outside

        labels = None if after.labels is None else after.labels[keep]
        return BoundingBoxes(clipped[keep], after.height, after.width, labels=labels)

    def wrap_data(self, data):
        """
        Wraps the data into a ``DataContainer``, and copies it if the stream must not modify the input arrays.

        Parameters
        ----------
        data : DataContainer or dict or numpy.ndarray
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Data to apply the transforms to

        """
        data = BaseTransform.wrap_data(data)
        if not self.in_place:
            data = data.copy()
        return data

    @staticmethod
    def exec_stream(transforms, data, optimize_stack, shuffle=False):
        """
        Static method, executes the list of transformations for a given data point.

        Parameters
        ----------
        transforms : list
            List of transformations to execute
        data : DataContainer or dict
            Data to be augmented. See ``solt.data.DataContainer.from_dict``
            to check how a conversion from dict is done.
        optimize_stack : bool
            Whether to execute augmentations stack optimization.
        shuffle : bool
            Whether to apply the transforms in a random order.

        Returns
        -------
        out : DataContainer
            Result
        """

        data = BaseTransform.wrap_data(data)
        if shuffle:
            # A shuffled list, the order of the stream itself is kept
            transforms = random.sample(transforms, len(transforms))
        # Performing the transforms using the optimized stack
        if optimize_stack:
            transforms = Stream.optimize_transforms_stack(transforms, data)
        for trf in transforms:
            if isinstance(trf, BaseTransform):
                if not optimize_stack:
                    data = trf(data, return_torch=False)
                else:
                    data = trf.apply(data)
            elif isinstance(trf, Stream):
                if type(trf).__call__ is Stream.__call__:
                    # Nested streams leave the final processing, e.g. of the bounding boxes, to the outermost stream
                    data = trf._apply(trf.wrap_data(data))
                else:
                    # A stream with its own __call__ must still run its logic
                    data = trf(data, return_torch=False)
            else:
                raise TypeError("Unknown transform type found in the Stream!")
        return data


class SelectiveStream(Stream):
    """
    Stream that uniformly selects n out of k given transforms.

    """

    serializable_name = "selective_stream"
    """How the class should be stored in the registry"""

    def __init__(
        self,
        transforms=None,
        n=1,
        probs=None,
        optimize_stack=False,
        ignore_fast_mode=False,
        in_place=True,
        max_bbox_area_loss=0.9,
        max_bbox_outside=0.5,
    ):
        """
        Constructor.

        Parameters
        ----------
        transforms : list
            List of k transforms to sample from
        n : int or tuple of int
            How many transforms to sample. If tuple (or list), the number of transforms is chosen uniformly
            from its values at every call, e.g. ``n=(1, 3)`` applies either 1 or 3 transforms.
            Every value must be between 0 and the number of transforms.
        probs : list of float
            Probabilities of the transforms to sample. If None, the transforms will be sampled uniformly.
        optimize_stack : bool
            Whether to execute stack optimization for augmentations.
        ignore_fast_mode : bool
            Whether to ignore the fast mode. This option enables full geometric transforms.
        in_place : bool
            Whether the transforms may modify the input arrays in place. See ``Stream``.
        max_bbox_area_loss : float or None
            Maximal fraction of the area, which the bounding boxes may lose. See ``Stream``.
        max_bbox_outside : float or None
            Maximal fraction of the area of the bounding boxes outside the image. See ``Stream``.
        """
        super(SelectiveStream, self).__init__(
            transforms=transforms,
            optimize_stack=optimize_stack,
            ignore_fast_mode=ignore_fast_mode,
            in_place=in_place,
            max_bbox_area_loss=max_bbox_area_loss,
            max_bbox_outside=max_bbox_outside,
        )
        if transforms is None:
            transforms = []
        if isinstance(n, (tuple, list)):
            n = tuple(n)
            if len(n) == 0:
                raise ValueError("Tuple of numbers of transforms to sample must not be empty")
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in n):
                raise TypeError("Tuple of numbers of transforms to sample must contain only ints")
        if any(x < 0 or x > len(transforms) for x in (n if isinstance(n, tuple) else (n,))):
            raise ValueError("Number of transforms to sample must be between 0 and the number of transforms")
        if probs is not None:
            if len(probs) != len(transforms):
                raise ValueError("Number of probabilities must be equal to the number of transforms")
        self.n = n
        self.probs = probs

    def _apply(self, data):
        """
        Applies randomly selected n transforms to the given data container.

        Parameters
        ----------
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result

        """
        if len(self.transforms) > 0:
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            if isinstance(self.n, tuple):
                n = random_state.choice(self.n)
            else:
                n = self.n

            trfs = random_state.choice(self.transforms, n, replace=False, p=self.probs)
            if self.optimize_stack:
                # The optimization fuses the transforms into the first one, so the selected ones are copied.
                # exec_stream optimizes them, so this must happen only once.
                trfs = [copy.deepcopy(x) for x in trfs]
            data = Stream.exec_stream(trfs, data, self.optimize_stack)
        return data
