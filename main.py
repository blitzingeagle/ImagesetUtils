from load_imageset import load_imageset
from resize_image import resize_to_mean

from operator import mul
import numpy as np
import cv2


def columnize(dataset):
    return [elem.reshape(reduce(mul, elem.shape, 1)) for elem in dataset]


if __name__ == "__main__":
    input_dir = "images"
    imageset = resize_to_mean(load_imageset("images", cv2.IMREAD_GRAYSCALE))
    data = columnize(imageset)

    print(np.array(data).shape)
