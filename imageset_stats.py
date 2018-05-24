from __future__ import division
from __future__ import print_function

from load_imageset import load_imageset

from pprint import pprint


def imageset_stats(imageset):
    stats = {}
    count = len(imageset)
    shape = {}

    height_sum, width_sum = 0, 0
    for img in imageset:
        height_sum += img.shape[0]
        width_sum += img.shape[1]
    shape["mean_height"] = height_sum / count
    shape["mean_width"] = width_sum / count

    stats["count"] = count
    stats["shape"] = shape

    return stats


if __name__ == "__main__":
    directory = "images"
    images = load_imageset(directory)
    stats = imageset_stats(images)
    pprint(stats, width=1)
