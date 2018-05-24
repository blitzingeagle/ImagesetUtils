import numpy as np
import cv2

import os.path as path
from glob import glob

def image_paths(dir):
    return sorted(glob(path.join(dir, "*")))


if __name__ == "__main__":
    paths = image_paths("images")
    print(paths)