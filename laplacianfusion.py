# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:26:15 2016

@author: Rachid & Chaima
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import image
import utils 

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0 
        return c

class LaplacianMap(object):
    """Class for weights attribution with Laplacian Fusion"""
    
    def __init__(self, fmt, names):
        """names is a liste of names, fmt is the format of the images"""
        self.images = []
        for name in names:
            self.images.append(image.Image(fmt, name))
        self.shape = self.images[0].shape
        self.num_images = len(self.images)
        
    def get_weights_map(self, w_c = 1, w_s = 1, w_e = 1):
        """Return the normalized Weight map"""
        self.weights = []
        sums = np.zeros((self.shape[0], self.shape[1]))
        for image_name in self.images:
            contrast = image_name.contrast()
            saturation = image_name.saturation()
            exposedness = image_name.exposedness()
            weight = (contrast**w_c)*(saturation**w_s)*(exposedness**w_e)
            self.weights.append(weight)
            sums = sums + weight
        for index in range(self.num_images):
            self.weights[index] = div0(self.weights[index],sums)
        return self.weights  
        
    def get_gaussian_pyramid_weights(self, n=3):
        """Return the Gaussian Pyramid of the Weight map"""
        weights = self.get_weights_map()
        self.weights_pyramid = []
        for index in range(self.num_images):
            print index
            weight_pyramid_floors = []
            for floor in range(n):
                print floor
                weight_pyramid_floors.append(utils.Reduce(weights[index],floor))
            self.weights_pyramid.append(weight_pyramid_floors)
        return self.weights_pyramid 