# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import image


class WeightsMap(object):
    """Class for weights attribution for all images"""

    def __init__(self, fmt, names):
        """names is a liste of names, fmt is the format of the images"""
        self.images = []
        for name in names:
            self.images.append(image.Image(fmt, name))

    def get_weights_map(self):
        self.weights = []
        w_c = 1
        w_s = 1
        w_e = 1
        for image_name in self.images:
            contrast = image_name.contrast()
            saturation = image_name.saturation()
            exposedness = image_name.exposedness()
            weight = (contrast**w_c)*(saturation**w_s)*(exposedness**w_e)
            self.weights.append(weight)
        return self.weights
