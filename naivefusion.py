# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import image

#def div0( a, b ):
#    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
#    with np.errstate(divide='ignore', invalid='ignore'):
#        c = np.true_divide( a, b )
#        c[ ~ np.isfinite( c )] = 0 
#        return c
        
class WeightsMap(object):
    """Class for weights attribution for all images"""

    def __init__(self, fmt, names):
        """names is a liste of names, fmt is the format of the images"""
        self.images = []
        for name in names:
            self.images.append(image.Image(fmt, name))
        self.shape = self.images[0].shape
        self.num_images = len(self.images)

    def get_weights_map(self, w_c, w_s, w_e):
        """Return the normalized Weight map"""
        self.weights = []
        sums = np.zeros((self.shape[0], self.shape[1]))
        for image_name in self.images:
            contrast = image_name.contrast()
            saturation = image_name.saturation()
            exposedness = image_name.exposedness()
            weight = (contrast**w_c)*(saturation**w_s)*(exposedness**w_e) + 1e-12
#            weight = ndimage.gaussian_filter(weight, sigma=(3, 3), order=0)
            self.weights.append(weight)
            sums = sums + weight
        for index in range(self.num_images):
            self.weights[index] = self.weights[index]/sums
        return self.weights   
    
    def result_exposure(self, w_c = 1, w_s = 1, w_e = 1):
        "Return the Exposure Fusion image with Naive method"
        self.get_weights_map(w_c, w_s, w_e)
        self.result_image = np.zeros(self.shape)
        for canal in range(3):
            for index in range(self.num_images):
                self.result_image[:,:,canal] += self.weights[index] * self.images[index].array[:,:,canal]
        self.result_image[self.result_image<0] = 0
        self.result_image[self.result_image>1] = 1
        return self.result_image

if __name__ == "__main__":
    names = [line.rstrip('\n') for line in open('list_jpeg_test.txt')]
    W = WeightsMap("mountain", names)
    im = W.result_exposure(1,0,0)
    image.show(im)
    misc.imsave("res/mountain_contr_naif.jpg", im)