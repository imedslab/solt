import os
import pandas as pd
import cv2
import random
import numpy as np

from tqdm import tqdm
from timeit import Timer
from collections import defaultdict
from augbench import utils
from augbench import transforms

os.environ["OMP_NUM_THREADS"] = "1"  # noqa E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa E402
os.environ["MKL_NUM_THREADS"] = "1"  # noqa E402
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # noqa E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa E402


cv2.setNumThreads(0)  # noqa E402
cv2.ocl.setUseOpenCL(False)  # noqa E402


if __name__ == "__main__":
    args = utils.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    package_versions = utils.get_package_versions()
    if args.print_package_versions:
        print(package_versions)
    images_per_second = defaultdict(dict)
    libraries = args.libraries
    data_dir = args.data_dir
    paths = list(sorted(os.listdir(data_dir)))
    paths = paths[: args.images]
    imgs_cv2 = [utils.read_img_cv2(os.path.join(data_dir, path), args.imsize) for path in paths]
    imgs_pillow = [utils.read_img_pillow(os.path.join(data_dir, path), args.imsize) for path in paths]

    benchmarks = [
        transforms.HorizontalFlip(args.imsize),
        transforms.VerticalFlip(args.imsize),
        transforms.RotateAny(args.imsize),
        transforms.Crop(224, args.imsize),
        transforms.Crop(128, args.imsize),
        transforms.Crop(64, args.imsize),
        transforms.Crop(32, args.imsize),
        transforms.Pad(300, args.imsize),
        transforms.VHFlipRotateCrop(args.imsize),
        transforms.HFlipCrop(args.imsize),
    ]
    print(f"==> Setting deterministic to be {args.deterministic}")
    for b in benchmarks:
        b.set_deterministic(args.deterministic)

    for library in libraries:
        imgs = imgs_pillow if library in ("torchvision", "augmentor", "pillow") else imgs_cv2
        pbar = tqdm(total=len(benchmarks))

        for benchmark in benchmarks:
            pbar.set_description("Current benchmark: {} | {}".format(library, benchmark))
            benchmark_images_per_second = None
            if benchmark.is_supported_by(library):
                timer = Timer(lambda: benchmark.run(library, imgs))
                run_times = timer.repeat(number=1, repeat=args.runs)
                benchmark_images_per_second = [1 / (run_time / args.images) for run_time in run_times]
            images_per_second[library][str(benchmark)] = benchmark_images_per_second
            pbar.update(1)
        pbar.close()
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: utils.format_results(r, args.show_std))
    df = df[libraries]
    augmentations = [str(i) for i in benchmarks]
    df = df.reindex(augmentations)
    if args.markdown:
        makedown_generator = utils.MarkdownGenerator(df, package_versions)
        makedown_generator.print()
    else:
        print(df.head(len(augmentations)))
