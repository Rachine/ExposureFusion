# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:32:39 2016

@author: Rachid & Chaima  
"""
import pdb

import numpy as np
import scipy.signal as sig


def kernel_1D(n, a=0.6):
    """Kernel function in 1 dimension"""
    kernel = [.0625, .25, .375, .25, .0625]
#    if n == 0:
#        return a
#    elif n == -1 or n == 1:
#        return 1./4
#    elif n == -2 or n == 2:
#        return 1./4 - float(a)/2
#    else:
#        return 0
    return kernel[n]


def kernel_old(m, n, a=0.6):
    """Returns the value of the kernel at position w and n"""
    return kernel_1D(m, a)*kernel_1D(n, a)


def get_kernel(a=0.6):
    kernel = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            kernel[i, j] = kernel_1D(i, a)*kernel_1D(j, a)
    return kernel


def Reduce_old(image, n, a=0.6):
    """Reduce function for Pyramids"""
    try:
        if n == 0:
            return image
        else:
            image = Reduce(image, n-1, a)
            [R, C] = [image.shape[0], image.shape[1]]
            image_extended = np.zeros((R+4, C+4))
            image_extended[2:R+2, 2:C+2] = image
            try:
                image_reduced = np.zeros((R/2, C/2))
            except Exception as e:
                print "Dimension Error"
                print e
            
            for i in range(R/2):
                for j in range(C/2):
                    for m in range(-2, 3):
                        for n in range(-2, 3):
                            image_reduced[i, j] += kernel_old(m, n) * image_extended[2 * i + m + 2, 2 * j + n + 2]
            return image_reduced
    except Exception as e:
        print "Dimension Error"
        print e


def weighted_sum(image, i, j, a):
    weighted_sum = 0
    for m in range(-2, 3):
        for n in range(-2, 3):
            pixel_i = float(i - m) / 2
            pixel_j = float(j - n) / 2
            if pixel_i.is_integer() and pixel_j.is_integer():
                weighted_sum += kernel_old(m, n, a) * image[pixel_i, pixel_j]
    return 4 * weighted_sum


def Expand_old(image, n, a=0.6):
    """Expand function for Pyramids"""
    try:
        if n == 0:
            return image
        else:
            image = Expand(image, n-1, a)
            [R, C] = image.shape
            image_extended = np.zeros((R+4, C+4))
            image_extended[2:R+2, 2:C+2] = image
            new_floor = np.zeros((2 * R, 2 * C))
            for i in range(2 * R):
                for j in range(2 * C):
                    new_floor[i, j] = weighted_sum(image_extended, i+2, j+2, a)
            new_floor = (new_floor - np.min(new_floor))
            new_floor = new_floor / np.max(new_floor)
            return new_floor                
    except Exception as e:
        print "Dimension error"
        print e


def Reduce1(image, a=0.6):
    kernel = get_kernel(a)
    shape = image.shape
    if len(shape) == 3:
        image_reduced = np.zeros((shape[0]/2, shape[1]/2, 3))
        for canal in range(3):
            canal_reduced = sig.convolve2d(image[:, :, canal], kernel, 'same')
            image_reduced[:, :, canal] = canal_reduced[::2, ::2]
    else:
        image_reduced = sig.convolve2d(image, kernel, 'same')[::2, ::2]
    return image_reduced


def Reduce(image, n, a=0.6):
    """Reduce function for Pyramids"""
    try:
        if n == 0:
            return image
        else:
            image = Reduce(image, n-1, a)
            return Reduce1(image, a)
    except Exception as e:
        print "Dimension Error"
        print e


def Expand1(image, a=0.6):
    kernel = get_kernel(a)
    shape = image.shape
    if len(shape) == 3:
        image_to_expand = np.zeros((2*shape[0], 2*shape[1], 3))
        image_expanded = np.zeros(image_to_expand.shape)
        for canal in range(3):
            image_to_expand[::2, ::2, canal] = image[:, :, canal]
            image_expanded[:, :, canal] = sig.convolve2d(image_to_expand[:, :, canal], 4*kernel, 'same')
    else:
        image_to_expand = np.zeros((2 * shape[0], 2 * shape[1]))
        image_to_expand[::2, ::2] = image
        image_expanded = sig.convolve2d(image_to_expand[:, :], 4*kernel, 'same')
    return image_expanded


def Expand(image, n, a=0.6):
    """Expand function for Pyramids"""
    try:
        if n == 0:
            return image
        else:
            image = Expand(image, n-1, a)
            return Expand1(image, a)
    except Exception as e:
        print "Dimension error"
        print e
