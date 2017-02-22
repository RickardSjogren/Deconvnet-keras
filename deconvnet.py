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
from keras.layers import (
    Input,
    InputLayer,
    Flatten,
    Activation,
    Dense)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D)


class DLayer:

    def __init__(self, layer):
        self.layer = layer
        self.up_data = None
        self.down_data = None
        self.up_func = None
        self.down_func = None

    def up(self, data, learning_phase=0):
        """
        function to compute activation in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        """
        data = self.up_func([data, learning_phase])
        self.up_data = data if K.backend() == 'theano' else data[0]
        return self.up_data

    def down(self, data, learning_phase=0):
        """
        function to compute activation in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Activation
        """
        data = self.down_func([data, learning_phase])
        self.down_data = data if K.backend() == 'theano' else data[0]
        return self.down_data


class DConvolution2D(DLayer):
    """
    A class to define forward and backward operation on Convolution2D
    """

    def __init__(self, layer):
        """
        # Arguments
            layer: an instance of Convolution2D layer, whose configuration 
                   will be used to initiate DConvolution2D(input_shape, 
                   output_shape, weights)
        """
        super(DConvolution2D, self).__init__(layer)

        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Set up_func for DConvolution2D
        input = Input(shape=layer.input_shape[1:])

        output = Convolution2D(
            nb_filter=layer.nb_filter,
            nb_row=layer.nb_row,
            nb_col=layer.nb_col,
            border_mode=layer.border_mode,
            weights=[W, b]
        )(input)
        self.up_func = _K_function([input, K.learning_phase()], output)

        # Flip W horizontally and vertically, 
        # and set down_func for DConvolution2D
        if K.image_dim_ordering() == 'th':
            W = np.transpose(W, (1, 0, 2, 3))
            W = W[:, :, ::-1, ::-1]
            nb_down_filter = W.shape[0]
            nb_down_row = W.shape[2]
            nb_down_col = W.shape[3]
        else:
            W = np.transpose(W, (0, 1, 3, 2))
            W = W[::-1, ::-1, :, :]
            nb_down_filter = W.shape[3]
            nb_down_row = W.shape[0]
            nb_down_col = W.shape[1]
        b = np.zeros(nb_down_filter)
        input = Input(shape=layer.output_shape[1:])
        output = Convolution2D(
            nb_filter=nb_down_filter,
            nb_row=nb_down_row,
            nb_col=nb_down_col,
            border_mode='same',
            weights=[W, b]
        )(input)
        self.down_func = _K_function([input, K.learning_phase()], output)


class DDense(DLayer):
    """
    A class to define forward and backward operation on Dense
    """

    def __init__(self, layer):
        """
        # Arguments
            layer: an instance of Dense layer, whose configuration 
                   will be used to initiate DDense(input_shape, 
                   output_shape, weights)
        """
        super(DDense, self).__init__(layer)
        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Set up_func for DDense
        input = Input(shape=layer.input_shape[1:])
        output = Dense(output_dim=layer.output_shape[1],
                       weights=[W, b])(input)
        self.up_func = _K_function([input, K.learning_phase()], output)

        # Transpose W and set down_func for DDense
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        input = Input(shape=self.output_shape[1:])
        output = Dense(
            output_dim=self.input_shape[1],
            weights=flipped_weights)(input)
        self.down_func = _K_function([input, K.learning_phase()], output)


class DPooling(DLayer):
    """
    A class to define forward and backward operation on Pooling
    """

    def __init__(self, layer):
        """
        # Arguments
            layer: an instance of Pooling layer, whose configuration 
                   will be used to initiate DPooling(input_shape, 
                   output_shape, weights)
        """
        super(DPooling, self).__init__(layer)
        self.poolsize = layer.pool_size

    def up(self, data, learning_phase=0):
        """
        function to compute pooling output in forward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Pooled result
        """
        [self.up_data, self.switch] = \
            self.__max_pooling_with_switch(data, self.poolsize)
        return self.up_data

    def down(self, data, learning_phase=0):
        """
        function to compute unpooling output in backward pass
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Unpooled result
        """
        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return self.down_data

    def __max_pooling_with_switch(self, input, poolsize):
        """
        Compute pooling output and switch in forward pass, switch stores 
        location of the maximum value in each poolsize * poolsize block
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
        # Returns
            Pooled result and Switch
        """
        switch = np.zeros(input.shape)

        if K.image_dim_ordering() == 'th':
            samples, dims, rows, cols = input.shape
        else:
            samples, rows, cols, dims = input.shape
        
        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        rows = rows // row_poolsize
        cols = cols // col_poolsize
        
        if K.image_dim_ordering() == 'th':
            out_shape = samples, dims, rows, cols
        else:
            out_shape = samples, rows, cols, dims
        
        pooled = np.zeros(out_shape)

        for sample in range(samples):
            for dim in range(dims):
                for row in range(rows):
                    for col in range(cols):
                        if K.image_dim_ordering() == 'th':
                            patch = input[sample,
                                    dim,
                                    row * row_poolsize: (row + 1) * row_poolsize,
                                    col * col_poolsize: (col + 1) * col_poolsize]
                            max_value = patch.max()
                            pooled[sample, dim, row, col] = max_value
                        else:
                            patch = input[sample,
                                    row * row_poolsize: (row + 1) * row_poolsize,
                                    col * col_poolsize: (col + 1) * col_poolsize,
                                    dim]
                            max_value = patch.max()
                            pooled[sample, row, col, dim] = max_value

                        max_col_index = patch.argmax(axis=1)
                        max_cols = patch.max(axis=1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        if K.image_dim_ordering() == 'th':
                            switch[sample,
                                   dim,
                                   row * row_poolsize + max_row,
                                   col * col_poolsize + max_col] = 1
                        else:
                            switch[sample,
                                   row * row_poolsize + max_row,
                                   col * col_poolsize + max_col, 
                                   dim] = 1
                            
        return [pooled, switch]

    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, input, switch):
        """
        Compute unpooled output using pooled data and switch
        # Arguments
            input: data to be pooled
            poolsize: size of pooling operation
            switch: switch storing location of each elements
        # Returns
            Unpooled result
        """
        if K.image_dim_ordering() == 'th':
            row_i, col_i = 2, 3
        else:
            row_i, col_i = 1, 2

        tile = np.ones((switch.shape[row_i] // input.shape[row_i],
                        switch.shape[col_i] // input.shape[col_i]))

        if K.image_dim_ordering() == 'th':
            out = np.kron(input, tile)
        else:
            out = np.kron(np.transpose(input, (0, 3, 1, 2)), tile)
            out = np.transpose(out, (0, 2, 3, 1))

        unpooled = out * switch
        return unpooled


class DActivation(DLayer):
    """
    A class to define forward and backward operation on Activation
    """

    def __init__(self, layer, linear=False):
        """
        # Arguments
            layer: an instance of Activation layer, whose configuration 
                   will be used to initiate DActivation(input_shape, 
                   output_shape, weights)
        """
        super(DActivation, self).__init__(layer)
        self.linear = linear
        self.activation = layer.activation
        input = K.placeholder(shape=layer.output_shape)

        output = self.activation(input)
        # According to the original paper, 
        # In forward pass and backward pass, do the same activation(relu)
        self.up_func = _K_function(
            [input, K.learning_phase()], output)
        self.down_func = _K_function(
            [input, K.learning_phase()], output)


class DFlatten(DLayer):
    """
    A class to define forward and backward operation on Flatten
    """

    def __init__(self, layer):
        """
        # Arguments
            layer: an instance of Flatten layer, whose configuration 
                   will be used to initiate DFlatten(input_shape, 
                   output_shape, weights)
        """
        super(DFlatten, self).__init__(layer)
        self.shape = layer.input_shape[1:]
        self.up_func = _K_function(
            [layer.input, K.learning_phase()], layer.output)

    def down(self, data, learning_phase=0):
        """
        function to unflatten input in backward pass
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            Recovered data
        """
        new_shape = [data.shape[0]] + list(self.shape)
        assert np.prod(self.shape) == np.prod(data.shape[1:])
        self.down_data = np.reshape(data, new_shape)
        return self.down_data


class DInput(DLayer):
    """
    A class to define forward and backward operation on Input
    """
    def up(self, data, learning_phase=0):
        """
        function to operate input in forward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in forward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            data
        """
        self.up_data = data
        return self.up_data

    def down(self, data, learning_phase=0):
        """
        function to operate input in backward pass, the input and output
        are the same
        # Arguments
            data: Data to be operated in backward pass
            learning_phase: learning_phase of Keras, 1 or 0
        # Returns
            data
        """
        self.down_data = data
        return self.down_data
    
    
def _K_function(inputs, out):
    if K.backend() == 'theano' or isinstance(out, (list, tuple)):
        return K.function(inputs, out)
    else:
        return K.function(inputs, [out])


def visualize(model, data, layer_name, feature_to_visualize, visualize_mode):
    """
    function to visualize feature
    # Arguments
        model: Pre-trained model used to visualize data
        data: image to visualize
        layer_name: Name of layer to visualize
        feature_to_visualize: Featuren to visualize
        visualize_mode: Visualize mode, 'all' or 'max', 'max' will only pick 
                        the greates activation in a feature map and set others
                        to 0s, this will indicate which part fire the neuron 
                        most; 'all' will use all values in a feature map,
                        which will show what image the filter sees. For 
                        convolutional layers, There is difference between 
                        'all' and 'max', for Dense layer, they are the same
    # Returns
        The image reflecting feature
    """
    deconv_layers = []
    # Stack layers
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], Convolution2D):
            deconv_layers.append(DConvolution2D(model.layers[i]))
            deconv_layers.append(
                DActivation(model.layers[i]))
        elif isinstance(model.layers[i], MaxPooling2D):
            deconv_layers.append(DPooling(model.layers[i]))
        elif isinstance(model.layers[i], Dense):
            deconv_layers.append(DDense(model.layers[i]))
            deconv_layers.append(
                DActivation(model.layers[i]))
        elif isinstance(model.layers[i], Activation):
            deconv_layers.append(DActivation(model.alyers[i]))
        elif isinstance(model.layers[i], Flatten):
            deconv_layers.append(DFlatten(model.layers[i]))
        elif isinstance(model.layers[i], InputLayer):
            deconv_layers.append(DInput(model.layers[i]))
        else:
            print('Cannot handle this type of layer')
            print(model.layers[i].get_config())
            sys.exit()
        if layer_name == model.layers[i].name:
            break

    # Forward pass
    deconv_layers[0].up(data)
    for i in range(1, len(deconv_layers)):
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    output = deconv_layers[-1].up_data
    assert output.ndim == 2 or output.ndim == 4
    if output.ndim == 2:
        feature_map = output[:, feature_to_visualize]
    else:
        feature_map = output[:, feature_to_visualize, :, :]
    if 'max' == visualize_mode:
        max_activation = feature_map.max()
        temp = feature_map == max_activation
        feature_map = feature_map * temp
    elif 'all' != visualize_mode:
        print('Illegal visualize mode')
        sys.exit()
    output = np.zeros_like(output)
    if 2 == output.ndim:
        output[:, feature_to_visualize] = feature_map
    else:
        output[:, feature_to_visualize, :, :] = feature_map

    # Backward pass
    deconv_layers[-1].down(output)
    for i in range(len(deconv_layers) - 2, -1, -1):
        deconv_layers[i].down(deconv_layers[i + 1].down_data)
    deconv = deconv_layers[0].down_data
    deconv = deconv.squeeze()

    return deconv


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
                        choices=['max', 'all'], default='all',
                        help='Visualize mode, \'max\' mode will pick the greatest \
                    activation in the feature map and set others to zero, \
                    \'all\' mode will use all values in the feature map')
    return parser


def main():
    import sys
    sys.path.insert(0,
                    r'C:\Users\risj0005\Documents\Kod\deep-learning-models-master')
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

    deconv = visualize(model, img_array,
                       layer_name, feature_to_visualize, visualize_mode)

    # postprocess and save image
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
