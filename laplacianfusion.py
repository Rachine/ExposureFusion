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
#
#def div0( a, b ):
#    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
#    with np.errstate(divide='ignore', invalid='ignore'):
#        c = np.true_divide( a, b )
#        c[ ~ np.isfinite( c )] = 0 
#        return c

class LaplacianMap(object):
    """Class for weights attribution with Laplacian Fusion"""
    
    def __init__(self, fmt, names, n=3):
        """names is a liste of names, fmt is the format of the images"""
        self.images = []
        for name in names:
            self.images.append(image.Image(fmt, name, crop=True, n=n))
        self.shape = self.images[0].shape
        self.num_images = len(self.images)
        self.height_pyr = n

    def get_weights_map(self, w_c, w_s, w_e):
        """Return the normalized Weight map"""
        self.weights = []
        sums = np.zeros((self.shape[0], self.shape[1]))
        for image_name in self.images:
            contrast = image_name.contrast()
            saturation = image_name.saturation()
            exposedness = image_name.exposedness()
            weight = (contrast**w_c)*(saturation**w_s)*(exposedness**w_e) + 1e-12
            self.weights.append(weight)
            sums = sums + weight
        for index in range(self.num_images):
            self.weights[index] = self.weights[index]/sums
        return self.weights

    def get_gaussian_pyramid(self,image, n):
        """Return the Gaussian Pyramid of an image"""
        gaussian_pyramid_floors = [image]
        for floor in range(1,n):
            gaussian_pyramid_floors.append(utils.Reduce(gaussian_pyramid_floors[-1],1))
        return gaussian_pyramid_floors
        
    def get_gaussian_pyramid_weights(self):
        """Return the Gaussian Pyramid of the Weight map of all images"""
        self.weights_pyramid = []
        for index in range(self.num_images):
            self.weights_pyramid.append(self.get_gaussian_pyramid(self.weights[index],self.height_pyr))
        return self.weights_pyramid
    
    def get_laplacian_pyramid(self,image, n):
        """Return the Laplacian Pyramid of an image"""
        gaussian_pyramid_floors = self.get_gaussian_pyramid(image,n)
        laplacian_pyramid_floors = [gaussian_pyramid_floors[-1]]
        for floor in range(n-2,-1,-1):
            new_floor =gaussian_pyramid_floors[floor] - utils.Expand(gaussian_pyramid_floors[floor+1],1)
            laplacian_pyramid_floors = [new_floor] + laplacian_pyramid_floors
        return laplacian_pyramid_floors 
    
    def get_laplacian_pyramid_images(self):
        """Return all the Laplacian pyramid for all images"""
        self.laplacian_pyramid = []
        for index in range(self.num_images):
            self.laplacian_pyramid.append(self.get_laplacian_pyramid(self.images[index].array,self.height_pyr))
        return self.laplacian_pyramid
    
    def result_exposure(self,  w_c = 1, w_s = 1, w_e = 1):
        "Return the Exposure Fusion image with Laplacian/Gaussian Fusion method"
        print "weights"
        self.get_weights_map(w_c, w_s, w_e)
        print "gaussian pyramid"
        self.get_gaussian_pyramid_weights()
        print "laplacian pyramid"
        self.get_laplacian_pyramid_images()
        result_pyramid = []
        for floor in range(self.height_pyr):
            print 'floor ', floor
            result_floor = np.zeros(self.laplacian_pyramid[0][floor].shape)
            for index in range(self.num_images):
                print 'image ', index
                for canal in range(3):
                    result_floor[:,:,canal] += self.laplacian_pyramid[index][floor][:,:,canal]*self.weights_pyramid[index][floor]
            result_pyramid.append(result_floor)
        # Get the image from the Laplacian pyramid
        self.result_image = result_pyramid[-1]
        for floor in range(self.height_pyr -2, -1, -1):
            print 'floor ', floor
            self.result_image = result_pyramid[floor] + utils.Expand(self.result_image,1)
        self.result_image[self.result_image<0] = 0
        self.result_image[self.result_image>1] = 1
        return self.result_image

    
if __name__ == "__main__":
    names = [line.rstrip('\n') for line in open('list_jpeg_test.txt')]
    lap = LaplacianMap('mountain',names,n=6)
    res = lap.result_exposure(1,1,1)
    image.show(res)