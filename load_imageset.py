import cv2

import os.path as path
from glob import glob


def image_paths(directory):
    return sorted(glob(path.join(directory, "*")))


def load_images(filepaths, option=cv2.IMREAD_COLOR):
    return [cv2.imread(filepath, option) for filepath in filepaths]


def load_imageset(directory, option=cv2.IMREAD_COLOR):
    return load_images(image_paths(directory), option)

