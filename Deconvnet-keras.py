#!/usr/bin/env python
# coding=utf-8
###############################################
# File Name: DeconvNet2D.py
# Author: Liang Jiang
# mail: jiangliang0811@gmail.com
# Created Time: Sun 30 Oct 2016 09:52:15 PM CST
# Description: Code for Deconvnet based on keras
###############################################
import keras.backend as K

import numpy as np
from PIL import Image

import deconvnet
from deconvnet.layers import *
from keras.layers import *


def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path of image to visualize')
    parser.add_argument('--layer_name', '-l',
                        action='store', dest='layer_name',
                        default='block5_conv1', help='Layer to visualize')
    parser.add_argument('--feature', '-f',
                        action='store', dest='feature',
                        default=0, type=int, help='Feature to visualize')
    parser.add_argument('--mode', '-m', action='store', dest='mode',
                        choices=['max', 'all'], default='max',
                        help='Visualize mode, \'max\' mode will pick the greatest \
                    activation in the feature map and set others to zero, \
                    \'all\' mode will use all values in the feature map')
    return parser


def main():
    import sys
    import vgg16
    import imagenet_utils
    # from keras.applications import vgg16, imagenet_utils
    parser = argparser()
    args = parser.parse_args()
    image_path = args.image
    layer_name = args.layer_name
    feature_to_visualize = args.feature
    visualize_mode = args.mode

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if not layer_name in layer_dict:
        print('Wrong layer name')
        sys.exit()

    # Load data and preprocess
    img = Image.open(image_path)
    img_array = np.array(img)
    if K.image_dim_ordering() == 'th':
        img_array = np.transpose(img_array, (2, 0, 1))
    img_array = img_array[np.newaxis, :]
    img_array = img_array.astype(np.float)
    img_array = imagenet_utils.preprocess_input(img_array)

    deconv_network = deconvnet.model.DeconvNetModel(model)
    deconv = deconv_network.deconvolve_feature_map(
        img_array, layer_name, feature_to_visualize, visualize_mode
    )

    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8)
    if K.image_dim_ordering() == 'th':
        deconv = np.transpose(deconv, (1, 2, 0))

    deconv = deconv[:, :, ::-1]
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    img.save('results/{}_{}_{}.png'.format(layer_name, feature_to_visualize,
                                           visualize_mode))


if "__main__" == __name__:
    main()
