import cv2
import os
import sys
import numpy as np


def show_img(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_img(img):
    cv2.imwrite(os.path.join(sys.path[0], "new_img.jpg"), emptyImage)


def salt_effect(img, n):
    if img.ndim == 2:
        for k in range(n):
            x = int(np.random.random() * img.shape[1])
            y = int(np.random.random() * img.shape[0])

            img[y, x] = 255
    else:
        for k in range(n):
            x = int(np.random.random() * img.shape[1])
            y = int(np.random.random() * img.shape[0])

            img[y, x, 0] = 255
            img[y, x, 1] = 255
            img[y, x, 2] = 255

    return img


def show_rgb(img):
    b, g, r = cv2.split(img)

    cv2.imshow("Blue", r)
    cv2.imshow("Red", g)
    cv2.imshow("Green", b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobel_operation(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return absX, absY, dst


def img_resize(img, new_size):
    img = cv2.resize(img, (int(img.shape[1] * new_size), int(img.shape[0] * new_size)))

    return img


if __name__ == "__main__":

    path = os.path.join(sys.path[0], "menu.jpg")

    img = cv2.imread(path)

    img = img_resize(img, 0.25)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_2 = cv2.GaussianBlur(img, (3, 3), 1.5)

    img_2 = cv2.Canny(img_2, 150, 50)

    ret, img_2 = cv2.threshold(img_2, 127, 255, cv2.THRESH_BINARY)

    print("ret: ", ret)

    show_img(img_2)


