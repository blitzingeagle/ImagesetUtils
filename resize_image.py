from __future__ import print_function

from load_imageset import load_imageset
from imageset_stats import imageset_stats

import numpy as np
import cv2

import os.path as path


def resize_images(images, shape=(128.128)):
    return [cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC) for img in images]


if __name__ == "__main__":
    input_dir = "images"
    output_dir = "data"

    imageset = load_imageset(input_dir)
    imgset_stats = imageset_stats(imageset)

    target_shape = (int(imgset_stats["shape"]["mean_width"]), int(imgset_stats["shape"]["mean_height"]))
    print("Target shape:", target_shape)

    outputset = resize_images(imageset, shape=target_shape)
    for idx, img in enumerate(outputset):
        cv2.imwrite(path.join(output_dir, "%05d.png" % idx), img)
