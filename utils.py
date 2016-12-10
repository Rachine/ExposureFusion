# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:32:39 2016

@author: Rachid & Chaima  
"""
import pdb

import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc

def kernel_1D(n, a=0.6):
    """Kernel function in 1 dimension"""
    if n == 0:
        return a
    elif n == -1 or n == 1:
        return 1./4
    elif n == -2 or n == 2:
        return 1./4 - float(a)/2
    else:
        return 0

def kernel(m, n, a=0.6):
    """Returns the value of the kernel at position w and n"""
    return kernel_1D(m,a)*kernel_1D(n, a)

def Reduce(image, a = 0.6):
    """Reduce function for Pyramids"""
    [R,C] = [image.shape[0],image.shape[1]]
    image_extended = np.zeros((R+4, C+4))
    image_extended[2:R+2,2:C+2]=image
    try:
        image_reduced = np.zeros((R/2,C/2))
    except Exception as e:
        print "Dimension Error"
        print e
    
    for i in range(R/2):
        for j in range(C/2):
            for m in range(-2,3):
                for n in range(-2,3):
                    image_reduced[i,j] += kernel(m,n)*image_extended[2*i+m+2,2*j+n+2]
    return image_reduced
    

def weighted_sum(image, i, j, a):
    weighted_sum = 0
    for m in range(-2, 3):
        for n in range(-2, 3):
            pixel_i = float(i - m) / 2
            pixel_j = float(j - n) / 2
            if pixel_i.is_integer() and pixel_j.is_integer():
                weighted_sum += kernel(m, n, a) * image[pixel_i, pixel_j]
    return 4 * weighted_sum


def Expand(image, n, a=0.6):
    """Expand function for Pyramids"""
    try:
        if n == 0:
            return image
        else:
            image = Expand(image, n-1, a)
            [R, C] = image.shape
            image_extended = np.zeros((R+4, C+4))
            image_extended[2:R+2,2:C+2]=image
            new_floor = np.zeros((2 * R, 2 * C))
            for i in range(2 * R):
                for j in range(2 * C):
                    new_floor[i, j] = weighted_sum(image_extended, i+2, j+2, a)
            new_floor = (new_floor - np.min(new_floor))
            new_floor = new_floor/np.max(new_floor)
            return new_floor                
    except Exception as e:
        print "Dimension error"
        print e
        
        
        
        
        
        
        