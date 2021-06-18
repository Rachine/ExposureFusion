# encoding=utf-8

import os.path

# from scipy import misc
import imageio
import matplotlib.pyplot as plt
import numpy as np


def weightedAverage(pixel):
    """
    """
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


def exponential_euclidean(canal, sigma):
    return np.exp(-(canal - 0.5) ** 2 / (2 * sigma ** 2))


def show(color_array):
    """ Function to show image"""
    plt.imshow(color_array)
    plt.show()
    plt.axis('off')


def show_gray(gray_array):
    """ Function to show grayscale image"""
    fig = plt.figure()
    plt.imshow(gray_array, cmap=plt.cm.Greys_r)
    plt.show()
    plt.axis('off')


class Image(object):
    """Class for Image"""

    def __init__(self, fmt, path, crop=False, n=0):
        self.path = os.path.join("image_set", fmt, str(path))
        self.fmt = fmt
        self.array = imageio.imread(self.path)
        self.array = self.array.astype(np.float32) / 255
        if crop:
            self.crop_image(n)
        self.shape = self.array.shape

    def crop_image(self, n):
        """
        """
        resolution = 2 ** n
        (height, width, _) = self.array.shape
        (max_height, max_width) = (resolution * (height // resolution),
                                   resolution * (width // resolution))
        (begin_height, begin_width) = ((height - max_height) / 2,
                                       (width - max_width) / 2)
        self.array = self.array[int(begin_height):int(max_height + begin_height),
                     int(begin_width):int(max_width + begin_width)]

    @property
    def grayScale(self):
        """Grayscale image"""
        rgb = self.array
        self._grayScale = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return self._grayScale

    def saturation(self):
        """Function that returns the Saturation map"""
        red_canal = self.array[:, :, 0]
        green_canal = self.array[:, :, 1]
        blue_canal = self.array[:, :, 2]
        mean = (red_canal + green_canal + blue_canal) / 3.0
        saturation = np.sqrt(((red_canal - mean) ** 2 + (green_canal - mean) ** 2 +
                              (blue_canal - mean) ** 2) / 3)
        return saturation

    def contrast(self):
        """Function that returns the Constrast numpy array"""
        grey = self.grayScale
        contrast = np.zeros((self.shape[0], self.shape[1]))
        grey_extended = np.zeros((self.shape[0] + 2, self.shape[1] + 2))
        grey_extended[1:self.shape[0] + 1, 1:self.shape[1] + 1] = grey
        #        kernel = np.array([[ -1, -1, -1 ],
        #                           [ -1, 8, -1 ],
        #                            [ -1, -1, -1 ]])
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                contrast[row][col] = np.abs(
                    (kernel *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
        contrast = (contrast - np.min(contrast))
        contrast = contrast / np.max(contrast)

        return contrast

    def sobel(self):
        """Function that returns the Constrast numpy array"""
        grey = self.grayScale
        sobel_h = np.zeros((self.shape[0], self.shape[1]))
        sobel_v = np.zeros((self.shape[0], self.shape[1]))
        grey_extended = np.zeros((self.shape[0] + 2, self.shape[1] + 2))
        grey_extended[1:self.shape[0] + 1, 1:self.shape[1] + 1] = grey
        kernel1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernel2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                sobel_h[row][col] = np.abs(
                    (kernel1 *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
                sobel_v[row][col] = np.abs(
                    (kernel2 *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
        return sobel_h, sobel_v

    def exposedness(self):
        """Function that returns the Well-Exposedness map"""
        red_canal = self.array[:, :, 0]
        green_canal = self.array[:, :, 1]
        blue_canal = self.array[:, :, 2]
        sigma = 0.2
        red_exp = exponential_euclidean(red_canal, sigma)
        green_exp = exponential_euclidean(green_canal, sigma)
        blue_exp = exponential_euclidean(blue_canal, sigma)
        return red_exp * green_exp * blue_exp


if __name__ == "__main__":
    im = Image("jpeg", "grandcanal_mean.jpg")
    sat = im.contrast()
    show_gray(sat)
