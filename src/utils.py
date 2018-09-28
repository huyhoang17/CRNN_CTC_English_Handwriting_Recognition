import random

import cv2
import numpy as np

import src.confing as cf


def preprocess(img, img_size, data_aug=False):
    """
    Put img into target img of size img_size
    Transpose for TF and normalize gray-values
    """

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([img_size[1], img_size[0]])

    # increase dataset size by applying random stretches to the images
    if data_aug:
        stretch = (random.random() - 0.5)  # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)),
                         1)  # random width, but at least 1
        img = cv2.resize(
            img, (wStretched,
                  img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

    # create target image and copy sample image into it
    (wt, ht) = img_size
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    new_size = (max(min(wt, int(w / f)), 1), max(
        min(ht, int(h / f)),
        1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, new_size)
    target = np.ones([ht, wt]) * 255
    target[0:new_size[1], 0:new_size[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img


def train_test_split_file(train_size=0.95):
    no_lines = 0
    with open(cf.WORDS_DATA) as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            no_lines += 1
    count = 0
    index_split = int(train_size * no_lines)

    with open(cf.WORDS_DATA) as f, \
            open(cf.WORDS_TRAIN, "w") as f_train, \
            open(cf.WORDS_TEXT, "w") as f_test:
        for line in f:
            if not line or line.startswith('#'):
                continue

            if count < index_split:
                f_train.write(line.strip() + "\n")
                count += 1
            else:
                f_test.write(line.strip() + "\n")
                count += 1


if __name__ == '__main__':
    train_test_split_file(train_size=0.95)
