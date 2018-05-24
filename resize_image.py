import numpy as np
import cv2


def resize_images(images, dsize=(128.128)):
    return [cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC) for img in images]


if __name__ == "__main__":
    img = cv2.imread("test_input.png")[:,:,::-1]
    cv2.imshow("img", img)

    res = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("resized", res)

    cv2.waitKey()
    cv2.destroyAllWindows()

