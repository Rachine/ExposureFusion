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
import pdb

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
        """Return the Gaussian Pyramid of the Weight map of all images"""
        weights = self.get_weights_map()
        self.weights_pyramid = []
        for index in range(self.num_images):
            self.weights_pyramid.append(self.get_gaussian_pyramid(weights[index],n))
        return self.weights_pyramid 
        
    def get_gaussian_pyramid(self,image, n=3):
        """Return the Gaussian Pyramid of an image"""
        gaussian_pyramid_floors = [image]
        for floor in range(1,n):
            gaussian_pyramid_floors.append(utils.Reduce(gaussian_pyramid_floors[-1],1))
        return gaussian_pyramid_floors 
    
    def get_laplacian_pyramid(self,image, n=3):
        """Return the Laplacian Pyramid of an image"""
        gaussian_pyramid_floors = self.get_gaussian_pyramid(image,n)
        laplacian_pyramid_floors = [gaussian_pyramid_floors[-1]]
        for floor in range(n-2,-1,-1):
            new_floor =gaussian_pyramid_floors[floor] - utils.Expand(gaussian_pyramid_floors[floor+1],1)
            laplacian_pyramid_floors = [new_floor] + laplacian_pyramid_floors
        return laplacian_pyramid_floors 
    
    def get_laplacian_pyramid_images(self, n=3):
        """Return all the Laplacian pyramid for all images"""
        self.laplacian_pyramid = []
        for index in range(self.num_images):
            self.laplacian_pyramid.append(self.get_laplacian_pyramid(image.grayScale[index],n))
        return self.laplacian_pyramid
    
    def result_exposure(self):
        "Return the Exposure Fusion image with Laplacian/Gaussian Fusion method"
        self.get_weights_map()
        self.result_image = np.zeros(self.shape)
#        for canal in range(3):
#            for index in range(self.num_images):
#                self.result_image[:,:,canal] += self.weights[index] * self.images[index].array[:,:,canal]
        return self.result_image
    
    
    
names = [line.rstrip('\n') for line in open('list_jpeg_test.txt')]
lap = LaplacianMap('jpeg',names)
fl = lap.get_laplacian_pyramid(lap.images[0].grayScale,4)