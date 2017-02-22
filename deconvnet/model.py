import collections

import numpy as np
from keras.layers import Dense, Convolution2D, \
    MaxPooling2D, Flatten, InputLayer, Activation
from keras import backend as K

from . import layers


class DeconvNetModel:

    """ Convert convolutional `keras.models.Model` to DeconvNet.

    Parameters
    ----------
    model : keras.models.Model
        Convolutional neural network model.

    Attributes
    ----------
    layers : list
        List of deconv-layers.
    layers_by_name : dict
        Name to layer map of network DeconvNet-layers.
    """

    def __init__(self, model):
        self.layers = []

        # Stack layers
        for layer in model.layers:
            if isinstance(layer, Convolution2D):
                self.layers.append(layers.DConvolution2D(layer))
                self.layers.append(layers.DActivation(layer))
            elif isinstance(layer, MaxPooling2D):
                self.layers.append(layers.DPooling(layer))
            elif isinstance(layer, Dense):
                self.layers.append(layers.DDense(layer))
                self.layers.append(layers.DActivation(layer))
            elif isinstance(layer, Activation):
                self.layers.append(layers.DActivation(layer))
            elif isinstance(layer, Flatten):
                self.layers.append(layers.DFlatten(layer))
            elif isinstance(layer, InputLayer):
                self.layers.append(layers.DInput(layer))
            else:
                raise ValueError('Cannot handle this type of layer')

        names = (layer.name for layer in model.layers)
        self.layers_by_name = collections.OrderedDict(zip(names, self.layers))

    def predict(self, data, layer=None):
        """ Predict output from `data`.

        Parameters
        ----------
        data : array_like
            Input data.
        layer : int, str, deconvnet.layers.DLayer, optional
            Target layer. If None, run all layers.

        Returns
        -------
        np.ndarray
            Network prediction of `data`.
        """
        if layer is None:
            end_layer = self.layers[-1]
        elif isinstance(layer, int):
            end_layer = self.layers[layer]
        elif isinstance(layer, str):
            end_layer = self.layers_by_name[layer]
        else:
            raise ValueError('invalid layer')

        output = self.layers[0].up(data)
        for layer in self.layers[1:]:
            output = layer.up(output)

            if layer is end_layer:
                break

        return output

    def deconvolve(self, output, layer):
        """ Deconvolve network output.

        Parameters
        ----------
        output : array_like
            Network output features from `layer`.
        layer : int, str, deconvnet.layers.DLayer
            Target layer.

        Returns
        -------
        np.ndarray
            Deconvolved `output`.
        """
        if isinstance(layer, int):
            layer = self.layers[layer]
        elif isinstance(layer, str):
            layer = self.layers_by_name[layer]

        if layer not in self.layers:
            raise ValueError('invalid layer')

        reversed_layers = list(reversed(self.layers))
        layer_i = reversed_layers.index(layer)

        deconv = reversed_layers[layer_i].down(output)
        for layer in reversed_layers[layer_i + 1:]:
            deconv = layer.down(deconv)

        return deconv.squeeze()

    def deconvolve_feature_map(self, data, layer, feature_map, mode):
        """ Deconvolve target feature.

        Parameters
        ----------
        data : array_like
            (n x m) Input data.
        layer : int, str, deconvnet.layers.DLayer
            Target layer.
        feature_map : int
            Index of feature map to use.
        mode : {'max', 'all'}
            Filtering mode of feature-map.

        Returns
        -------
        np.ndarray
            (n x m) Deconvolved feature-map.
        """
        output = self.predict(data, layer)
        if output.ndim == 2:
            feature = output[:, feature_map]
        else:
            if K.image_dim_ordering() == 'th':
                feature = output[:, feature_map, :, :]
            else:
                feature = output[:, :, :, feature_map]

        if mode == 'max':
            max_activation = feature.max()
            temp = feature == max_activation
            feature = feature * temp
        elif mode == 'all':
            pass
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        output = np.zeros_like(output)
        if 2 == output.ndim:
            output[:, feature_map] = feature
        else:
            if K.image_dim_ordering() == 'th':
                output[:, feature_map, :, :] = feature
            else:
                output[:, :, :, feature_map] = feature

        return self.deconvolve(output, layer)
