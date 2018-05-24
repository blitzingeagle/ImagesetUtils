from load_imageset import load_imageset
from resize_image import resize_to_mean

from operator import mul
import numpy as np


def columnize(dataset):
    return [elem.reshape(reduce(mul, elem.shape, 1)) for elem in dataset]


if __name__ == "__main__":
    input_dir = "images"
    imageset = resize_to_mean(load_imageset("images"))
    data = columnize(imageset)

    print(np.array(data).shape)
