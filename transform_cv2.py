import cv2
import numbers
import numpy as np


def crop_center(img, cx, cy, crop_w, crop_h):
    img_h, img_w = img.shape[:2]
    trunc_top = trunc_bottom = trunc_left = trunc_right = 0
    left = cx - crop_w//2
    if left < 0:
        trunc_left = 0 - left
        left = 0
    right = left - trunc_left + crop_w
    if right > img_w:
        trunc_right = right - img_w
        right = img_w
    top = cy - crop_h//2
    if top < 0:
        trunc_top = 0 - top
        top = 0
    bottom = top - trunc_top + crop_h
    if bottom > img_h:
        trunc_bottom = bottom - img_h
        bottom = img_h
    if trunc_left or trunc_right or trunc_top or trunc_bottom:
        img_new = np.zeros((crop_h, crop_w, img.shape[2]), dtype=img.dtype)
        trunc_bottom = crop_h - trunc_bottom
        trunc_right = crop_w - trunc_right
        img_new[trunc_top:trunc_bottom, trunc_left:trunc_right] = img[top:bottom, left:right]
        return img_new
    else:
        return img[top:bottom, left:right]


class ScaleCv(object):
    """Rescales the input OpenCv image
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.shape[:2]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return cv2.resize(img, (ow, oh), self.interpolation)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return cv2.resize(img, (ow, oh), self.interpolation)


class CenterCropCv(object):
    """Crops the given OpenCv image
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = self.size
        x1 = int(round(w / 2.))
        y1 = int(round(h / 2.))
        return crop_center(img, x1, y1, th, tw)
