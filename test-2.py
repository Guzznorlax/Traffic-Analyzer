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


def show_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)

    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color=(255, 255, 0))

    return histImg


if __name__ == "__main__":

    size_th = 200

    path = os.path.join(sys.path[0], "input.mp4")

    video = cv2.VideoCapture(path)

    width, height = video.get(3), video.get(4)

    bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    kernalOp = np.ones((3, 3), np.uint8)
    kernalCl = np.ones((11, 11), np.uint)

    while(video.isOpened()):
        retval, frame = video.read()

        fg_mask = bg_subtractor.apply(frame)

        ret, th = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        mask_1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernalOp)

        mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_CLOSE, kernalCl)

        contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > (height * width / size_th):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.rectangle(mask_1, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.rectangle(th, (x, y), (x + w, y + h), (255, 255, 0), 2)

        frame_bin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hist = show_hist(frame_bin)

        cv2.imshow("main", frame)
        cv2.imshow("threshold", th)
        cv2.imshow("mask_1", mask_1)
        cv2.imshow("hist", frame_hist)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()