from __future__ import print_function

import numpy as np
import cv2

import os.path as path
from glob import glob

def image_paths(dir):
    return sorted(glob(path.join(dir, "*")))

def load_images(filepaths):
    return [cv2.imread(path) for path in filepaths]

if __name__ == "__main__":
    directory = "images"
    paths = image_paths(directory)
    N = len(paths)
    print(N, "images found.")

    images = load_images(paths)
    print(images)
