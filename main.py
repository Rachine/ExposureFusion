# coding: utf-8

import argparse
import image, naivefusion, laplacianfusion

###### Loading the arguments ######

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--list', dest='names', type=str, default='list_images.txt',
                    help='The text file which contains the names of the images')
parser.add_argument('-f', '--folder', dest='folder', type=str, required=True,
                    help='The folder containing the images')
parser.add_argument('-hp', '--heightpyr', dest='height_pyr', type=int, default=6,
                    help='The height of the Laplacian pyramid')
parser.add_argument('-wc', dest='w_c', type=float, default=1.0, help='Exponent of the contrast')
parser.add_argument('-ws', dest='w_s', type=float, default=1.0, help='Exponent of the saturation')
parser.add_argument('-we', dest='w_e', type=float, default=1.0, help='Exponent of the exposedness')
args = parser.parse_args()
params = vars(args) # convert to ordinary dict

names = [line.rstrip('\n') for line in open(params['names'])]
folder = params['folder']
height_pyr = params['height_pyr']
w_c = params['w_c']
w_s = params['w_s']
w_e = params['w_e']

###### Naive Fusion ######

W = naivefusion.WeightsMap(folder, names)
res_naive = W.result_exposure(w_c, w_s, w_e)
image.show(res_naive)

###### Laplacian Fusion ######

lap = laplacianfusion.LaplacianMap(folder, names, n=height_pyr)
res_lap = lap.result_exposure(w_c, w_s, w_e)
image.show(res_lap)
