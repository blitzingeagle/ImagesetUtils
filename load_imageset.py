import cv2

import os.path as path
from glob import glob


def image_paths(directory):
    return sorted(glob(path.join(directory, "*")))


def load_images(filepaths):
    return [cv2.imread(filepath) for filepath in filepaths]


def load_imageset(directory):
    return load_images(image_paths(directory))

