import numpy as np
import cv2

if __name__ == "__main__":
    img = cv2.imread("test_input.png")
    cv2.imshow("Image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

