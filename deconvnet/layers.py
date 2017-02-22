import numpy as np

from keras.layers import (
    Input,
    Dense)
from keras.layers.convolutional import (
    Convolution2D)
from keras import backend as K


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
                                    row * row_poolsize: (
                                                        row + 1) * row_poolsize,
                                    col * col_poolsize: (
                                                        col + 1) * col_poolsize]
                            max_value = patch.max()
                            pooled[sample, dim, row, col] = max_value
                        else:
                            patch = input[sample,
                                    row * row_poolsize: (
                                                        row + 1) * row_poolsize,
                                    col * col_poolsize: (
                                                        col + 1) * col_poolsize,
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