"""
models.py - Deep Learning Models for Seismic Phase Picking and Earthquake Detection

This module provides a collection of neural network architectures for 1D time series analysis, primarily focused on seismic phase picking and earthquake detection.
The models are based on variations of the PhaseNet architecture, with extensions including attention mechanisms, transformers, residual and depthwise convolutions, and multi-head outputs.
The code is built on TensorFlow and Keras layers.

Main Components:
----------------
- Utility Functions:
    - crop_and_concat: Center-crops or zero-pads tensors along the time axis and concatenates them.
    - crop_and_add: Center-crops or zero-pads tensors and adds them, with optional channel projection.

- Core Layers and Blocks:
    - TransformerBlock: Implements a multi-head self-attention transformer block for 1D data.
    - ResnetBlock1D: Standard 1D residual block with batch normalization and dropout.
    - DepthwiseResnetBlock1D: 1D residual block using depthwise separable convolutions.
    - ResidualConv1D: Stacked 1D residual convolutional layers with gated activations.

- Model Architectures:
    - PhaseNet: U-Net-like encoder-decoder for per-sample classification (e.g., phase picking).
    - EPick: PhaseNet variant with multi-head attention and residual attention connections.
    - TransPhaseNet: PhaseNet variant with transformer-based attention blocks and flexible RNN/conv context.
    - DepthwiseTransPhaseNet: Like TransPhaseNet but uses depthwise separable convolutions for efficiency.
    - SplitOutputTransPhaseNet: Produces separate outputs for P and S phase probabilities.
    - DepthwiseSplitOutputTransPhaseNet: Combines depthwise convolutions with split output for P and S phases.
    - SplitOutputTransPhaseNetBranch: Variant with shared decoder blocks before branching into P/S heads.
    - SplitOutputBranchDepthwise: Combines depthwise convolutions with branch-after-N decoder sharing.

Usage:
------
Import the desired model class and instantiate it with appropriate parameters. Example:

    from models import PhaseNet
    model = PhaseNet(num_classes=3, filters=[8,16,32,64], kernelsizes=[7,7,5,3])
    model.build((None, 1024, 3))
    model.summary()

References:
-----------
- PhaseNet: https://github.com/wayneweiqiang/PhaseNet
- EPick: https://arxiv.org/abs/2109.02567
- TPhaseNet: https://doi.org/10.1093/gji/ggae298

Author: Erik Myklebust, Andreas Koehler, Tord Stangeland, Steffen Mæland
License: MIT
"""

import tensorflow as tf 
import tensorflow.keras.layers as tfl 
import tensorflow.keras.backend as K
import numpy as np

def crop_and_concat(x, y):
    """
    Center-crop or zero-pad two tensors along the time axis and concatenate them.

    This function aligns two tensors along the time axis (axis 1) by center-cropping or zero-padding
    the second tensor as needed, then concatenates them along the channel axis.

    Parameters
    ----------
    x : tf.Tensor
        First input tensor of shape (batch, time, channels).
    y : tf.Tensor
        Second input tensor of shape (batch, time, channels).

    Returns
    -------
    tf.Tensor
        Concatenated tensor with aligned time axis.
    """
    to_crop = x.shape[1] - y.shape[1]
    if to_crop < 0:
        to_crop = abs(to_crop)
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.Cropping1D((of_start, of_end))(y)
    elif to_crop > 0:
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.ZeroPadding1D((of_start, of_end))(y)
    return tfl.concatenate([x,y])

def crop_and_add(x, y):
    """
    Add two tensors after center-cropping or zero-padding the time axis.

    This function aligns two tensors along the time axis (axis 1) by center-cropping or zero-padding
    the second tensor as needed, then adds them. If the channel dimensions differ and are statically
    known, the second tensor is projected to match the first.

    Parameters
    ----------
    x : tf.Tensor
        First input tensor of shape (batch, time, channels).
    y : tf.Tensor
        Second input tensor of shape (batch, time, channels).

    Returns
    -------
    tf.Tensor
        Sum of the two tensors with aligned time and channel axes.
    """

    # --- Temporal alignment (axis-1) ------------------------------------------------
    to_crop = x.shape[1] - y.shape[1]
    if to_crop < 0:
        # y is longer – crop it in the middle
        to_crop = abs(to_crop)
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.Cropping1D((of_start, of_end))(y)
    elif to_crop > 0:
        # y is shorter – pad both sides with zeros
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.ZeroPadding1D((of_start, of_end))(y)

    # --- Channel alignment (axis-2) -------------------------------------------------
    x_ch = K.int_shape(x)[2]
    y_ch = K.int_shape(y)[2]
    if (x_ch is not None) and (y_ch is not None) and (x_ch != y_ch):
        # Project y to the same number of channels as x only when both are
        # statically known and different.  If shapes are symbolic we skip the
        # projection, assuming the upstream graph ensures compatibility.
        y = tfl.Conv1D(x_ch, 1, padding='same')(y)

    return x + y


class TransformerBlock(tfl.Layer):
    """
    Transformer block for 1D data with multi-head self-attention and feed-forward network.

    This layer implements a transformer block as described in Vaswani et al. (2017),
    adapted for 1D time series data. It supports both legacy and modern call signatures
    for flexible integration into Keras models.

    Parameters
    ----------
    key_dim : int
        Size of each attention head for query and key.
    num_heads : int
        Number of attention heads.
    ff_dim : int
        Hidden layer size in the feed-forward network.
    value_dim : int, optional
        Size of each attention head for value. If None, defaults to key_dim.
    rate : float, optional
        Dropout rate for attention and feed-forward layers. Default is 0.1.

    Attributes
    ----------
    att : keras.layers.MultiHeadAttention
        Multi-head attention layer.
    ffn : keras.Sequential
        Feed-forward network.
    layernorm1 : keras.layers.LayerNormalization
        Layer normalization after attention.
    layernorm2 : keras.layers.LayerNormalization
        Layer normalization after feed-forward.
    dropout1 : keras.layers.Dropout
        Dropout after attention.
    dropout2 : keras.layers.Dropout
        Dropout after feed-forward.

    References
    ----------
    Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017.
    """
    def __init__(self, key_dim, num_heads, ff_dim, value_dim=None, rate=0.1):
        """
        Initialize the TransformerBlock.

        Parameters
        ----------
        key_dim : int
            Size of each attention head for query and key.
        num_heads : int
            Number of attention heads.
        ff_dim : int
            Hidden layer size in the feed-forward network.
        value_dim : int, optional
            Size of each attention head for value. If None, defaults to key_dim.
        rate : float, optional
            Dropout rate for attention and feed-forward layers. Default is 0.1.
        """
        super().__init__()
        self.att = tfl.MultiHeadAttention(num_heads=num_heads,
                                          key_dim=key_dim,
                                          value_dim=value_dim)
        self.ffn = tf.keras.Sequential(
            [tfl.Dense(ff_dim, activation="relu"), tfl.Dense(key_dim)]
        )
        self.layernorm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(rate)
        self.dropout2 = tfl.Dropout(rate)

    def call(self, query, value=None, training=None):
        """
        Call the transformer block with flexible input signatures.

        Parameters
        ----------
        query : tf.Tensor or list/tuple of tf.Tensor
            Query tensor or a (query, value) tuple/list for legacy support.
        value : tf.Tensor, optional
            Value tensor. If None, defaults to query (self-attention).
        training : bool, optional
            Whether the call is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor after attention and feed-forward processing.
        """

        # Handle the legacy "list / tuple" call pattern
        if value is None and isinstance(query, (list, tuple)):
            if len(query) != 2:
                raise ValueError(
                    "When passing a list/tuple to TransformerBlock you must "
                    "provide exactly two tensors (query & value)")
            query, value = query

        # Self-attention fallback
        if value is None:
            value = query

        attn_output = self.att(query, value)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ResnetBlock1D(tfl.Layer):
    """
    1D Residual Block with Batch Normalization and Dropout.

    Implements a standard 1D residual block with two convolutional layers, batch normalization,
    activation, and dropout. Used for feature extraction in time series models.

    Parameters
    ----------
    filters : int
        Number of convolutional filters.
    kernelsize : int
        Size of the convolutional kernel.
    activation : str, optional
        Activation function to use. Default is 'linear'.
    dropout : float, optional
        Dropout rate. Default is 0.1.
    **kwargs : dict
        Additional keyword arguments for Conv1D layers.

    Attributes
    ----------
    projection : keras.layers.Conv1D
        1x1 convolution for input projection.
    conv1, conv2 : keras.layers.Conv1D
        Main convolutional layers.
    dropout1 : keras.layers.Dropout
        Dropout layer.
    bn1, bn2, bn3 : keras.layers.BatchNormalization
        Batch normalization layers.
    add : keras.layers.Add
        Addition layer for residual connection.
    relu : keras.layers.Activation
        Activation function.
    """
    def __init__(self,
                 filters,
                 kernelsize,
                 activation='linear',
                 dropout=0.1, **kwargs):
        """
        Initialize a 1D residual block.

        Parameters
        ----------
        filters : int
            Number of convolutional filters.
        kernelsize : int
            Size of the convolutional kernel.
        activation : str, optional
            Activation function to use. Default is 'linear'.
        dropout : float, optional
            Dropout rate. Default is 0.1.
        **kwargs : dict
            Additional keyword arguments for Conv1D layers.
        """
        super(ResnetBlock1D, self).__init__()
        self.filters = filters
        self.projection = tfl.Conv1D(filters, 1, padding='same', **kwargs)
        self.conv1 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.conv2 = tfl.Conv1D(filters, kernelsize, activation=None, padding='same', **kwargs)
        self.dropout1 = tfl.Dropout(dropout)
        self.bn1 = tfl.BatchNormalization()
        self.bn2 = tfl.BatchNormalization()
        self.bn3 = tfl.BatchNormalization()
        self.add = tfl.Add()
        self.relu = tfl.Activation(activation)

    def call(self, inputs, training=None):
        """
        Forward pass for the 1D residual block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, time, channels).
        training : bool, optional
            Whether the call is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor after residual block processing.
        """
        x = self.projection(inputs)
        fx = self.bn1(inputs)
        fx = self.conv1(fx)
        fx = self.bn2(fx)
        fx = self.relu(fx)
        fx = self.dropout1(fx, training=training)
        #fx = self.dropout1(fx)
        fx = self.conv2(fx)
        x = self.add([x, fx])
        x = self.bn3(x)
        x = self.relu(x)
        return x


class DepthwiseResnetBlock1D(tfl.Layer):
    """
    1D Depthwise Separable Residual Block.

    Implements a 1D residual block using depthwise separable convolutions for efficiency.
    Includes batch normalization, dropout, and optional initial convolution.

    Parameters
    ----------
    filters : int
        Number of output filters.
    depth_multiplier : int
        Depth multiplier for depthwise convolution.
    kernelsize : int
        Size of the convolutional kernel.
    apply_conv_before : bool, optional
        Whether to apply a Conv1D before depthwise convolutions. Default is True.
    activation : str, optional
        Activation function. Default is 'linear'.
    dropout : float, optional
        Dropout rate. Default is 0.1.
    **kwargs : dict
        Additional keyword arguments for Conv1D/DepthwiseConv1D layers.

    Attributes
    ----------
    firstconv : keras.layers.Conv1D
        Initial convolutional layer (optional).
    depthconv1, depthconv2 : keras.layers.DepthwiseConv1D
        Depthwise convolutional layers.
    pointwiseconv1, pointwiseconv2 : keras.layers.Conv1D
        Pointwise convolutional layers.
    dropout1 : keras.layers.Dropout
        Dropout layer.
    bn1, bn2, bn3, bn4 : keras.layers.BatchNormalization
        Batch normalization layers.
    add : keras.layers.Add
        Addition layer for residual connection.
    relu : keras.layers.Activation
        Activation function.
    """
    def __init__(self,
                 filters,
                 depth_multiplier,
                 kernelsize,
                 apply_conv_before=True,
                 activation='linear',
                 dropout=0.1, **kwargs):
        """
        Initialize a 1D depthwise separable residual block.

        Parameters
        ----------
        filters : int
            Number of output filters.
        depth_multiplier : int
            Depth multiplier for depthwise convolution.
        kernelsize : int
            Size of the convolutional kernel.
        apply_conv_before : bool, optional
            Whether to apply a Conv1D before depthwise convolutions. Default is True.
        activation : str, optional
            Activation function. Default is 'linear'.
        dropout : float, optional
            Dropout rate. Default is 0.1.
        **kwargs : dict
            Additional keyword arguments for Conv1D/DepthwiseConv1D layers.
        """
        super(DepthwiseResnetBlock1D, self).__init__()
        self.apply_conv_before = apply_conv_before
        self.firstconv = tfl.Conv1D(filters, 3, padding='same')
        self.depthconv1 = tfl.DepthwiseConv1D(kernelsize, depth_multiplier=depth_multiplier,  activation=None, padding='same', **kwargs)
        self.depthconv2 = tfl.DepthwiseConv1D(kernelsize, depth_multiplier=depth_multiplier, activation=None, padding='same', **kwargs)
        self.pointwiseconv1 = tfl.Conv1D(filters, 1, padding='same')
        self.pointwiseconv2 = tfl.Conv1D(filters, 1, padding='same')
        self.dropout1 = tfl.Dropout(dropout)
        self.bn1 = tfl.BatchNormalization()
        self.bn2 = tfl.BatchNormalization()
        self.bn3 = tfl.BatchNormalization()
        self.bn4 = tfl.BatchNormalization()
        self.add = tfl.Add()
        self.relu = tfl.Activation(activation)

    def call(self, inputs, training=None):
        """
        Forward pass for the 1D depthwise separable residual block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, time, channels).
        training : bool, optional
            Whether the call is in training mode.

        Returns
        -------
        tf.Tensor
            Output tensor after depthwise separable residual block processing.
        """
        if self.apply_conv_before:
            x = self.firstconv(inputs)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = inputs
        fx = self.depthconv1(x)
        fx = self.bn2(fx)
        fx = self.relu(fx)
        fx = self.dropout1(fx, training=training)
        #fx = self.dropout1(fx)
        fx = self.pointwiseconv1(fx)
        fx = self.bn3(fx)
        fx = self.relu(fx)
        fx = self.depthconv2(fx)
        fx = self.pointwiseconv2(fx)
        x = self.add([x, fx])
        x = self.bn4(x)
        x = self.relu(x)
        return x


class ResidualConv1D(tfl.Layer):
    """
    Stacked 1D Residual Convolutional Block with Gated Activations.

    Implements a stack of gated residual convolutional layers, optionally with causal (dilated) convolutions.
    Used for modeling temporal dependencies in time series data.

    Parameters
    ----------
    filters : int, optional
        Number of filters. Default is 32.
    kernel_size : int, optional
        Size of the convolutional kernel. Default is 3.
    stacked_layer : int, optional
        Number of stacked residual layers. Default is 1.
    activation : str, optional
        Activation function for the final output. Default is 'relu'.
    causal : bool, optional
        If True, use causal (dilated) convolutions. Default is False.

    Attributes
    ----------
    sigmoid_layers : list of keras.layers.Conv1D
        Sigmoid convolutional layers for gating.
    tanh_layers : list of keras.layers.Conv1D
        Tanh convolutional layers for gating.
    conv_layers : list of keras.layers.Conv1D
        1x1 convolutional layers for residual connections.
    shape_matching_layer : keras.layers.Conv1D
        1x1 convolution for input projection.
    add : keras.layers.Add
        Addition layer for residual connection.
    final_activation : keras.activations
        Final activation function.
    """

    def __init__(self,
                 filters=32,
                 kernel_size=3,
                 stacked_layer=1,
                 activation='relu',
                 causal=False):
        """
        Initialize a stacked 1D residual convolutional block.

        Parameters
        ----------
        filters : int, optional
            Number of filters. Default is 32.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stacked_layer : int, optional
            Number of stacked residual layers. Default is 1.
        activation : str, optional
            Activation function for the final output. Default is 'relu'.
        causal : bool, optional
            If True, use causal (dilated) convolutions. Default is False.
        """

        super(ResidualConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stacked_layer = stacked_layer
        self.causal = causal
        self.activation = activation

    def build(self, input_shape):
        """
        Build the internal layers for the residual convolutional block.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor (batch, time, channels).
        """
        self.sigmoid_layers = []
        self.tanh_layers = []
        self.conv_layers = []

        self.shape_matching_layer = tfl.Conv1D(self.filters, 1, padding = 'same')
        self.add = tfl.Add()
        self.final_activation = tf.keras.activations.get(self.activation)

        for dilation_rate in [2 ** i for i in range(self.stacked_layer)]:
            self.sigmoid_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate,
                           padding='causal' if self.causal else 'same',
                                    activation='sigmoid'))
            self.tanh_layers.append(
                tfl.Conv1D(self.filters, self.kernel_size, dilation_rate=dilation_rate,
                           padding='causal' if self.causal else 'same',
                                    activation='tanh'))
            self.conv_layers.append(tfl.Conv1D(self.filters, 1, padding='same'))

    def get_config(self):
        """
        Return the configuration of the layer for serialization.

        Returns
        -------
        dict
            Dictionary of layer configuration parameters.
        """
        return dict(name=self.name,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    stacked_layer=self.stacked_layer)

    def call(self, inputs):
        """
        Forward pass for the stacked 1D residual convolutional block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, time, channels).

        Returns
        -------
        tf.Tensor
            Output tensor after stacked residual convolutional processing.
        """
        out = self.shape_matching_layer(inputs)
        residual_output = out
        x = inputs
        for sl, tl, cl in zip(self.sigmoid_layers, self.tanh_layers, self.conv_layers):
            sigmoid_x = sl(x)
            tanh_x = tl(x)

            x = tfl.multiply([sigmoid_x, tanh_x])
            x = cl(x)
            residual_output = tfl.add([residual_output, x])

        return self.final_activation(self.add([out, x]))



class PhaseNet(tf.keras.Model):
    """
    U-Net-like 1D Encoder-Decoder Model for Seismic Phase Picking.

    Implements a 1D U-Net architecture for per-sample classification, such as seismic phase picking.
    Supports configurable convolution types, pooling, and output activations.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes. Default is 2.
    num_channels : int, optional
        Number of input channels. Default is None.
    filters : list of int, optional
        List of filter counts for each block. Default is [4, 8, 16, 32].
    kernelsizes : list of int, optional
        List of kernel sizes for each block. Default is [7, 7, 7, 7].
    output_activation : str, optional
        Activation function for the output layer. Default is 'linear'.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Kernel regularizer for convolutional layers. Default is None.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    pool_type : {'max', 'avg'}, optional
        Type of pooling layer. Default is 'max'.
    activation : str, optional
        Activation function for intermediate layers. Default is 'relu'.
    initializer : str or keras.initializers.Initializer, optional
        Weight initializer. Default is 'glorot_normal'.
    conv_type : {'default', 'seperable'}, optional
        Type of convolutional layer. Default is 'default'.
    name : str, optional
        Model name. Default is 'PhaseNet'.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the U-Net.
    model : keras.Model
        Full Keras model for inference and training.
    """
    def __init__(self,
                 num_classes=2,
                 num_channels=None,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 pool_type='max',
                 activation='relu',
                 initializer='glorot_normal',
                 conv_type='default',
                 name='PhaseNet'):
        """
        Initialize the PhaseNet model.

        Parameters
        ----------
        num_classes : int, optional
            Number of output classes. Default is 2.
        num_channels : int, optional
            Number of input channels. Default is None.
        filters : list of int, optional
            List of filter counts for each block. Default is [4, 8, 16, 32].
        kernelsizes : list of int, optional
            List of kernel sizes for each block. Default is [7, 7, 7, 7].
        output_activation : str, optional
            Activation function for the output layer. Default is 'linear'.
        kernel_regularizer : keras.regularizers.Regularizer, optional
            Kernel regularizer for convolutional layers. Default is None.
        dropout_rate : float, optional
            Dropout rate. Default is 0.2.
        pool_type : {'max', 'avg'}, optional
            Type of pooling layer. Default is 'max'.
        activation : str, optional
            Activation function for intermediate layers. Default is 'relu'.
        initializer : str or keras.initializers.Initializer, optional
            Weight initializer. Default is 'glorot_normal'.
        conv_type : {'default', 'seperable'}, optional
            Type of convolutional layer. Default is 'default'.
        name : str, optional
            Model name. Default is 'PhaseNet'.
        """
        super(PhaseNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.activation = activation

        if filters is None:
            self.filters = [4, 8, 16, 32]
        else:
            self.filters = filters

        if kernelsizes is None:
            self.kernelsizes = [7, 7, 7, 7]
        else:
            self.kernelsizes = kernelsizes
            
        if pool_type == 'max':
            self.pool_layer = tfl.MaxPooling1D
        else:
            self.pool_layer = tfl.AveragePooling1D
            
        if conv_type == 'seperable':
            self.conv_layer = tfl.SeparableConv1D
        else:
            self.conv_layer = tfl.Conv1D

    def _down_block(self, f, ks, x):
        """
        Downsampling block for the encoder path.

        Parameters
        ----------
        f : int
            Number of filters.
        ks : int
            Kernel size.
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor after downsampling.
        """
        x = self.conv_layer(f, 
                        ks, 
                        padding="same",
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializer)(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        x = self.pool_layer(4, 2, padding='same')(x)
        return x
    
    def _up_block(self, f, ks, x):
        """
        Upsampling block for the decoder path.

        Parameters
        ----------
        f : int
            Number of filters.
        ks : int
            Kernel size.
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor after upsampling.
        """
        x = self.conv_layer(f, 
                            ks, 
                            padding="same",
                            kernel_regularizer=self.kernel_regularizer,
                            kernel_initializer=self.initializer)(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        x = tfl.UpSampling1D(2)(x)
        return x
        

    def build(self, input_shape):
        """
        Build the PhaseNet model architecture.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor (batch, time, channels).
        """
        inputs = tf.keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = self.conv_layer(self.filters[0], 
                            self.kernelsizes[0],
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i, _ in enumerate(self.filters):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            skips.append(x)
            
        skips = skips[:-1]

        self.encoder = tf.keras.Model(inputs, x)
        
        for i in list(range(len(self.filters)))[::-1]:
            x = self._up_block(self.filters[i], self.kernelsizes[i], x)
            x = crop_and_concat(x, skips[i])

        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = self.conv_layer(self.filters[0], 
                            self.kernelsizes[0],
                            kernel_regularizer=self.kernel_regularizer,
                            padding="same",
                            name='exit')(x)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        """
        Total number of trainable parameters in the model.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        return self.model.summary()

    def call(self, inputs):
        """
        Forward pass for the PhaseNet model.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, time, channels).

        Returns
        -------
        tf.Tensor
            Output tensor after model processing.
        """
        return self.model(inputs)
    
class EPick(PhaseNet):
    """
    EPick: PhaseNet Variant with Multi-Head Attention and Residual Attention.

    Implements the EPick architecture for seismic phase picking, extending PhaseNet with
    multi-head attention and residual attention connections as described in [Pickering et al., 2021].

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes. Default is 2.
    output_layer : keras.Layer, optional
        Optional custom output layer. Default is None.
    filters : list of int, optional
        List of filter counts for each block. Default is None.
    kernelsizes : list of int, optional
        List of kernel sizes for each block. Default is None.
    output_activation : str, optional
        Activation function for the output layer. Default is 'linear'.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Kernel regularizer for convolutional layers. Default is None.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    att_type : str, optional
        Attention type ('additive', 'dot', or 'concat'). Default is 'additive'.
    activation : str, optional
        Activation function for intermediate layers. Default is 'relu'.
    pool_type : {'max', 'avg'}, optional
        Type of pooling layer. Default is 'max'.
    initializer : str or keras.initializers.Initializer, optional
        Weight initializer. Default is 'glorot_normal'.
    residual_attention : list of int, optional
        List of residual attention sizes. Default is [16, 16, 16, 16, 16].
    name : str, optional
        Model name. Default is 'EPick'.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the model.
    model : keras.Model
        Full Keras model for inference and training.

    References
    ----------
    Li, W., et al. (2021). "EPick: Multi-Class Attention-based U-shaped Neural Network for Earthquake Detection and Seismic Phase Picking."
    https://arxiv.org/abs/2109.02567
    """
    def __init__(self,
                 num_classes=2,
                 output_layer=None,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 att_type='additive',
                 num_heads=8,
                 activation='relu',
                 pool_type='max',
                 initializer='glorot_normal',
                 residual_attention=None,
                 name='EPick'):
        """
        https://arxiv.org/abs/2109.02567
        
        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            residual_attention (list: optional): list of residual attention sizes, one longer that filters. 
            att_type (str): dot or concat
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
        """
        super(EPick, self).__init__(num_classes=num_classes, 
                                              filters=filters, 
                                              kernelsizes=kernelsizes, 
                                              output_activation=output_activation, 
                                              kernel_regularizer=kernel_regularizer, 
                                              dropout_rate=dropout_rate,
                                              pool_type=pool_type, 
                                              activation=activation, 
                                              initializer=initializer, 
                                              name=name)

        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention
        self.num_heads = num_heads

    def _down_block(self, f, ks, x):
        x = tfl.Conv1D(f, ks, padding="same",
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializer,
                        )(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        x = self.pool_layer(4, strides=2, padding='same')(x)
        return x

    def _up_block(self, f, ks, x, upsample=True):
        x = tfl.Conv1DTranspose(f, ks, padding="same",
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_initializer=self.initializer,
                                )(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        if upsample:
            x = tfl.UpSampling1D(2)(x)
        return x
            

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = tfl.Conv1D(self.filters[0], self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='entry')(inputs)

        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)
        
        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for ks, f in zip(self.kernelsizes[1:], self.filters[1:]):
            x = self._down_block(f, ks, x) 
            skips.append(x)
        
        attentions = []
        for i, skip in enumerate(skips):
            if self.residual_attention[i] <= 0:
                att = skip
            elif i == 0:
                att = tfl.MultiHeadAttention(num_heads=self.num_heads, 
                                             key_dim=self.residual_attention[i],)(skip, skip, return_attention_scores=False)
            else:
                tmp = []
                z = skips[i]
                for j, skip2 in enumerate(skips[:i]):
                    if self.residual_attention[j] <= 0:
                        att = tfl.Conv1D(self.filters[j], 3, activation='relu', padding='same')(z)
                    else:
                        att = tfl.MultiHeadAttention(num_heads=self.num_heads, 
                                                     key_dim=self.residual_attention[j])(z, skip2, return_attention_scores=False)
                    tmp.append(att)
                att = tfl.Concatenate()(tmp)
            attentions.append(att)
            
        x = crop_and_concat(x, attentions[-1])
        self.encoder = tf.keras.Model(inputs, x)
            
        i = len(self.filters) - 1
        for f, ks in zip(self.filters[::-1][:-1], self.kernelsizes[::-1][:-1]):
            x = self._up_block(f, ks, x, upsample = i != 0)
            x = crop_and_concat(x, attentions[i-1])
            i -= 1
        
        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        elif self.output_layer is not None:
            outputs = self.output_layer(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

class TransPhaseNet(PhaseNet):
    """
    TransPhaseNet: PhaseNet Variant with Transformer-Based Attention.

    Extends PhaseNet with transformer-based attention blocks and flexible context modeling
    (LSTM, causal convolutions, or depthwise convolutions). Supports both downstep and self-attention.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes. Default is 2.
    filters : list of int, optional
        List of filter counts for each block. Default is None.
    kernelsizes : list of int, optional
        List of kernel sizes for each block. Default is None.
    output_activation : str, optional
        Activation function for the output layer. Default is 'linear'.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Kernel regularizer for convolutional layers. Default is None.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    initializer : str or keras.initializers.Initializer, optional
        Weight initializer. Default is 'glorot_normal'.
    residual_attention : list of int, optional
        List of residual attention sizes. Default is [16, 16, 16, 16].
    pool_type : {'max', 'avg'}, optional
        Type of pooling layer. Default is 'max'.
    att_type : {'downstep', 'across'}, optional
        Attention type. Default is 'across'.
    rnn_type : {'lstm', 'causal', 'depthwise', 'none'}, optional
        Type of context modeling. 'none' skips RNN and goes straight to Transformer. Default is 'lstm'.
    additive_att : bool, optional
        If True, use additive attention. Default is True.
    stacked_layer : int, optional
        Number of stacked residual layers for causal context. Default is 4.
    name : str, optional
        Model name. Default is 'TransPhaseNet'.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the model.
    model : keras.Model
        Full Keras model for inference and training.
    """
    def __init__(self,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 initializer='glorot_normal',
                 residual_attention=None,
                 pool_type='max',
                 att_type='across',
                 num_transformers=1,
                 num_heads=4,
                 rnn_type='lstm',
                 additive_att=True,
                 stacked_layer=4,
                 activation='relu',
                 name='TransPhaseNet'):
        """Adapted to 1D from https://keras.io/examples/vision/oxford_pets_image_segmentation/

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            residual_attention (list: optional): list of residual attention sizes, one longer that filters. 
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
            att_type (str, optional): if the attention should work during downstep or across (self attention). 
            num_heads (int, optional): number of attention heads in transformer blocks. Defaults to 4.
            rnn_type (str, optional): use "lstm" rnns, "causal" dilated conv, "depthwise" conv, or "none" to skip RNN.  
        """
        super(TransPhaseNet, self).__init__(num_classes=num_classes, 
                                              filters=filters, 
                                              kernelsizes=kernelsizes, 
                                              output_activation=output_activation, 
                                              kernel_regularizer=kernel_regularizer, 
                                              dropout_rate=dropout_rate,
                                              pool_type=pool_type, 
                                              activation=activation, 
                                              initializer=initializer, 
                                              name=name)
        self.att_type = att_type
        self.rnn_type = rnn_type
        self.stacked_layer = stacked_layer
        self.additive_att = additive_att
        self.num_transformers = num_transformers
        self.num_heads = num_heads
            
        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention
    
    def _down_block(self, f, ks, x):
        x = ResnetBlock1D(f, 
                        ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)    
        x = self.pool_layer(4, strides=2, padding="same")(x)
        return x
    
    def _up_block(self, f, ks, x):
        x = ResnetBlock1D(f, 
                        ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)
        x = tfl.UpSampling1D(2)(x)
        return x

    def _att_block(self, x, y, ra):
        if self.rnn_type == 'lstm':
            x = tfl.Bidirectional(tfl.LSTM(ra, return_sequences=True))(x)
        elif self.rnn_type == 'causal':
            x1 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(x)
            rev_x = tfl.Lambda(lambda t: tf.reverse(t, axis=[1]))(x)
            x2 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(rev_x)
            x2_rev = tfl.Lambda(lambda t: tf.reverse(t, axis=[1]))(x2)
            x = tfl.Concatenate(axis=-1)([x1, x2_rev])
        elif self.rnn_type == 'depthwise':
            # Use a lightweight depth-wise separable convolution stack to model
            # local context while keeping the receptive field comparable to the
            # dilated-causal path but with far fewer multiplications.
            # One DepthwiseResnetBlock1D already contains two depth-wise convs
            # and residual connections, so a single block is usually enough.
            x = DepthwiseResnetBlock1D(
                filters=ra,
                depth_multiplier=1,
                kernelsize=3,
                activation=self.activation,
                dropout=self.dropout_rate,
            )(x)
        elif self.rnn_type == 'none':
            # Skip RNN layer, go straight to Transformer
            pass
        else:
            raise NotImplementedError('rnn type:' + self.rnn_type + ' is not supported')
        x = tfl.Conv1D(ra, 1, padding='same')(x)

        att = TransformerBlock(
            num_heads=self.num_heads,
            key_dim=ra,
            ff_dim=ra * 4,
            rate=self.dropout_rate,
        )(x, y)
        if self.num_transformers > 1:
            for _ in range(1, self.num_transformers):
                att = TransformerBlock(num_heads=self.num_heads,
                            key_dim=ra,
                            ff_dim=ra*4,
                            rate=self.dropout_rate)(att)

        return att


    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])
        
        # Entry block
        
        x = ResnetBlock1D(self.filters[0], 
                          self.kernelsizes[0], 
                          activation=self.activation, 
                          dropout=self.dropout_rate)(inputs)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, skips[-1], self.residual_attention[i])
                if self.additive_att:
                    x += att
                else:
                    x = crop_and_add(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
            skips.append(x)

        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        
        for i in range(1, len(self.filters)):
            x = self._up_block(self.filters[::-1][i], self.kernelsizes[::-1][i], x)
            
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)

        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = tfl.Conv1D(self.filters[0], 
                       self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='exit')(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            # 5 classes is hard-coded for 3 lables (P,S and Noise) + P and S beam
            if self.num_classes == 5 :
                x1 = tfl.Conv1D(3,
                           1,
                           padding="same")(x)
                x2 = tfl.Conv1D(2,
                           1,
                           padding="same")(x)

                outputs1 = tfl.Activation(self.output_activation, dtype='float32',name='labels')(x1)
                outputs2 = tfl.Activation(self.output_activation, dtype='float32',name='beams')(x2)
                #outputs1 = tfl.Activation(self.output_activation, name='labels')(x1)
                #outputs2 = tfl.Activation(self.output_activation, name='beams')(x2)
                outputs={'labels': outputs1, 'beams': outputs2}
            else :
                x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
                outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)
        
        
class DepthwiseTransPhaseNet(PhaseNet):
    """
    DepthwiseTransPhaseNet: Efficient PhaseNet Variant with Depthwise Convolutions.

    Like TransPhaseNet, but uses depthwise separable convolutions for efficiency.
    Suitable for large-scale or resource-constrained applications.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    num_classes : int, optional
        Number of output classes. Default is 2.
    filters : list of int, optional
        List of filter counts for each block. Default is None.
    kernelsizes : list of int, optional
        List of kernel sizes for each block. Default is None.
    apply_conv_before_block : bool, optional
        Whether to apply a Conv1D before depthwise blocks. Default is True.
    output_activation : str, optional
        Activation function for the output layer. Default is 'linear'.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Kernel regularizer for convolutional layers. Default is None.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    initializer : str or keras.initializers.Initializer, optional
        Weight initializer. Default is 'glorot_normal'.
    residual_attention : list of int, optional
        List of residual attention sizes. Default is [16, 16, 16, 16].
    pool_type : {'max', 'avg'}, optional
        Type of pooling layer. Default is 'max'.
    att_type : {'downstep', 'across'}, optional
        Attention type. Default is 'across'.
    rnn_type : {'lstm', 'causal', 'depthwise', 'none'}, optional
        Type of context modeling. Default is 'lstm'.
    additive_att : bool, optional
        If True, use additive attention. Default is True.
    stacked_layer : int, optional
        Number of stacked residual layers for causal context. Default is 4.
    name : str, optional
        Model name. Default is 'TransPhaseNet'.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the model.
    model : keras.Model
        Full Keras model for inference and training.
    """
    def __init__(self,
                 num_channels,
                 num_classes=2,
                 filters=None,
                 kernelsizes=None,
                 apply_conv_before_block=True,
                 output_activation='linear',
                 kernel_regularizer=None,
                 dropout_rate=0.2,
                 initializer='glorot_normal',
                 residual_attention=None,
                 pool_type='max',
                 att_type='across',
                 num_transformers=1,
                 num_heads=4,
                 rnn_type='lstm',
                 additive_att=True,
                 stacked_layer=4,
                 activation='relu',
                 name='TransPhaseNet'):
        """Adapted to 1D from https://keras.io/examples/vision/oxford_pets_image_segmentation/

        Args:
            num_classes (int, optional): number of outputs. Defaults to 2.
            filters (list, optional): list of number of filters. Defaults to None.
            kernelsizes (list, optional): list of kernel sizes. Defaults to None.
            residual_attention (list: optional): list of residual attention sizes, one longer that filters. 
            output_activation (str, optional): output activation, eg., 'softmax' for multiclass problems. Defaults to 'linear'.
            kernel_regularizer (tf.keras.regualizers.Regualizer, optional): kernel regualizer. Defaults to None.
            dropout_rate (float, optional): dropout. Defaults to 0.2.
            initializer (tf.keras.initializers.Initializer, optional): weight initializer. Defaults to 'glorot_normal'.
            name (str, optional): model name. Defaults to 'PhaseNet'.
            att_type (str, optional): if the attention should work during downstep or across (self attention). 
            rnn_type (str, optional): use "lstm" rnns, "causal" dilated conv, "depthwise" conv, or "none" to skip RNN.  
        """
        super(DepthwiseTransPhaseNet, self).__init__(
                                              num_channels=num_channels,
                                              num_classes=num_classes, 
                                              filters=filters, 
                                              kernelsizes=kernelsizes, 
                                              output_activation=output_activation, 
                                              kernel_regularizer=kernel_regularizer, 
                                              dropout_rate=dropout_rate,
                                              pool_type=pool_type, 
                                              activation=activation, 
                                              initializer=initializer, 
                                              name=name)
        self.num_channels = num_channels
        self.att_type = att_type
        self.rnn_type = rnn_type
        self.stacked_layer = stacked_layer
        self.additive_att = additive_att
        self.num_transformers = num_transformers
        self.num_heads = num_heads
            
        if residual_attention is None:
            self.residual_attention = [16, 16, 16, 16]
        else:
            self.residual_attention = residual_attention
    
    def _down_block(self, f, ks, x):
        x = DepthwiseResnetBlock1D(
                        f,
                        depth_multiplier=(f // self.num_channels), 
                        kernelsize=ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)    
        x = self.pool_layer(4, strides=2, padding="same")(x)
        return x
    
    def _up_block(self, f, ks, x):
        x = ResnetBlock1D(f, 
                        ks, 
                        activation=self.activation, 
                        dropout=self.dropout_rate)(x)
        x = tfl.UpSampling1D(2)(x)
        return x

    def _att_block(self, x, y, ra):
        if self.rnn_type == 'lstm':
            x = tfl.Bidirectional(tfl.LSTM(ra, return_sequences=True))(x)
        elif self.rnn_type == 'causal':
            x1 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(x)
            rev_x = tfl.Lambda(lambda t: tf.reverse(t, axis=[1]))(x)
            x2 = ResidualConv1D(ra, 3, stacked_layer=self.stacked_layer, causal=True)(rev_x)
            x2_rev = tfl.Lambda(lambda t: tf.reverse(t, axis=[1]))(x2)
            x = tfl.Concatenate(axis=-1)([x1, x2_rev])
        elif self.rnn_type == 'depthwise':
            # Use a lightweight depth-wise separable convolution stack to model
            # local context while keeping the receptive field comparable to the
            # dilated-causal path but with far fewer multiplications.
            # One DepthwiseResnetBlock1D already contains two depth-wise convs
            # and residual connections, so a single block is usually enough.
            x = DepthwiseResnetBlock1D(
                filters=ra,
                depth_multiplier=1,
                kernelsize=3,
                activation=self.activation,
                dropout=self.dropout_rate,
            )(x)
        else:
            raise NotImplementedError('rnn type:' + self.rnn_type + ' is not supported')
        x = tfl.Conv1D(ra, 1, padding='same')(x)
        
        att = TransformerBlock(
            num_heads=self.num_heads,
            key_dim=ra,
            ff_dim=ra * 4,
            rate=self.dropout_rate,
        )(x, y)

        if self.num_transformers > 1:
            for _ in range(1, self.num_transformers):
                att = TransformerBlock(num_heads=self.num_heads,
                            key_dim=ra,
                            ff_dim=ra*4,
                            rate=self.dropout_rate)(att)
        
        return att

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])
        
        # Entry block
        
        x = DepthwiseResnetBlock1D(self.filters[0], 
                          (self.filters[0] // self.num_channels),
                          self.kernelsizes[0], 
                          activation=self.activation, 
                          dropout=self.dropout_rate)(inputs)

        skips = [x]
        
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, skips[-1], self.residual_attention[i])
                if self.additive_att:
                    x += att
                else:
                    x = crop_and_add(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
            skips.append(x)

        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)

        self.encoder = tf.keras.Model(inputs, x)
        ### [Second half of the network: upsampling inputs] ###
        
        for i in range(1, len(self.filters)):
            x = self._up_block(self.filters[::-1][i], self.kernelsizes[::-1][i], x)
            
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)

        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)
        
        #Exit block
        x = tfl.Conv1D(self.filters[0], 
                       self.kernelsizes[0],
                       strides=1,
                       kernel_regularizer=self.kernel_regularizer,
                       padding="same",
                       name='exit')(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Add a per-pixel classification layer
        if self.num_classes is not None:
            x = tfl.Conv1D(self.num_classes,
                           1,
                           padding="same")(x)
            outputs = tfl.Activation(self.output_activation, dtype='float32')(x)
        else:
            outputs = x

        # Define the model
        self.model = tf.keras.Model(inputs, outputs)

    @property
    def num_parameters(self):
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

    def summary(self):
        return self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

class SplitOutputTransPhaseNet(TransPhaseNet):
    """
    SplitOutputTransPhaseNet: Dual-Head PhaseNet Variant for P and S Phase Probabilities.

    Produces two separate output branches for P-phase and S-phase probabilities, each as a single-channel output.
    Useful for tasks requiring explicit separation of phase probabilities.

    Parameters
    ----------
    filters : list of int, optional
        List of filter counts for each block. Default is None.
    kernelsizes : list of int, optional
        List of kernel sizes for each block. Default is None.
    output_activation : str, optional
        Activation function for the output layer. Default is 'sigmoid'.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Kernel regularizer for convolutional layers. Default is None.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    initializer : str or keras.initializers.Initializer, optional
        Weight initializer. Default is 'glorot_normal'.
    residual_attention : list of int, optional
        List of residual attention sizes. Default is None.
    pool_type : {'max', 'avg'}, optional
        Type of pooling layer. Default is 'max'.
    att_type : {'downstep', 'across'}, optional
        Attention type. Default is 'across'.
    rnn_type : {'lstm', 'causal', 'depthwise', 'none'}, optional
        Type of context modeling. Default is 'lstm'.
    additive_att : bool, optional
        If True, use additive attention. Default is True.
    stacked_layer : int, optional
        Number of stacked residual layers for causal context. Default is 4.
    activation : str, optional
        Activation function for intermediate layers. Default is 'relu'.
    name : str, optional
        Model name. Default is 'SplitOutputTransPhaseNet'.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the model.
    model : keras.Model
        Full Keras model for inference and training.
    """
    def __init__(
        self,
        filters=None,
        kernelsizes=None,
        output_activation='sigmoid',
        kernel_regularizer=None,
        dropout_rate=0.2,
        initializer='glorot_normal',
        residual_attention=None,
        pool_type='max',
        att_type='across',
        num_transformers=1,
        num_heads=4,
        rnn_type='lstm',
        additive_att=True,
        stacked_layer=4,
        activation='relu',
        name='SplitOutputTransPhaseNet'
    ):
        super(SplitOutputTransPhaseNet, self).__init__(
            num_classes=None,  # let child manage final heads
            filters=filters,
            kernelsizes=kernelsizes,
            output_activation=output_activation,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=dropout_rate,
            pool_type=pool_type,
            activation=activation,
            initializer=initializer,
            residual_attention=residual_attention,
            att_type=att_type,
            rnn_type=rnn_type,
            additive_att=additive_att,
            stacked_layer=stacked_layer,
            num_transformers=num_transformers,
            num_heads=num_heads,
            name=name,
        )
        # Our child won't rely on parent's final decode
        # We'll manually build 2 separate decoders for P and S

    def build_encoder(self, inputs):
        """
        Build the encoder part of the network.
        Returns the bottleneck features and skip connections.
        """
        x = ResnetBlock1D(
            self.filters[0],
            self.kernelsizes[0],
            activation=self.activation,
            dropout=self.dropout_rate
        )(inputs)

        # Collect skip-connections
        skips = [x]

        # Down blocks (1..N-1)
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, skips[-1], self.residual_attention[i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
            skips.append(x)

        # Possibly final self-attention in the bottleneck
        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)

        return x, skips

    def build_decoder(self, x, skips, input_shape, name_suffix):
        """
        Build a decoder path from encoded features and skip connections.
        Returns the final output tensor.
        """
        # Up blocks
        for i in range(1, len(self.filters)):
            x = self._up_block(
                self.filters[::-1][i],
                self.kernelsizes[::-1][i],
                x
            )
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)

        # Fix any shape mismatch with the input
        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)

        # Final layers
        x = tfl.Conv1D(
            self.filters[0],
            self.kernelsizes[0],
            strides=1,
            kernel_regularizer=self.kernel_regularizer,
            padding="same",
            name=f'exit_{name_suffix}'
        )(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Single-channel output with sigmoid activation
        x = tfl.Conv1D(1, 1, padding="same")(x)
        return tfl.Activation('sigmoid', dtype='float32', name=name_suffix)(x)

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        # Build encoder
        x, skips = self.build_encoder(inputs)

        # Save the encoder model if needed
        self.encoder = tf.keras.Model(inputs, x, name="encoder")

        # Build P and S decoders
        p_out = self.build_decoder(x, skips, input_shape, name_suffix="p_head")
        s_out = self.build_decoder(x, skips, input_shape, name_suffix="s_head")

        # Build the final model with two outputs
        self.model = tf.keras.Model(inputs=inputs, outputs=[p_out, s_out], name=self.name)

    def call(self, inputs):
        return self.model(inputs)

# -----------------------------------------------------------------------------
#  DepthwiseSplitOutputTransPhaseNet: Depthwise + Split Output variant
# -----------------------------------------------------------------------------
class DepthwiseSplitOutputTransPhaseNet(DepthwiseTransPhaseNet):
    """
    DepthwiseSplitOutputTransPhaseNet: Efficient Split Output Model with Depthwise Convolutions.

    Combines the efficiency of depthwise separable convolutions with the dual-head output
    structure of SplitOutputTransPhaseNet for P and S phase detection.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    filters : list of int, optional
        List of filter counts for each block. Default is None.
    kernelsizes : list of int, optional
        List of kernel sizes for each block. Default is None.
    apply_conv_before_block : bool, optional
        Whether to apply a Conv1D before depthwise blocks. Default is True.
    output_activation : str, optional
        Activation function for the output layer. Default is 'sigmoid'.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Kernel regularizer for convolutional layers. Default is None.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    initializer : str or keras.initializers.Initializer, optional
        Weight initializer. Default is 'glorot_normal'.
    residual_attention : list of int, optional
        List of residual attention sizes. Default is [16, 16, 16, 16].
    pool_type : {'max', 'avg'}, optional
        Type of pooling layer. Default is 'max'.
    att_type : {'downstep', 'across'}, optional
        Attention type. Default is 'across'.
    rnn_type : {'lstm', 'causal', 'depthwise', 'none'}, optional
        Type of context modeling. Default is 'lstm'.
    additive_att : bool, optional
        If True, use additive attention. Default is True.
    stacked_layer : int, optional
        Number of stacked residual layers for causal context. Default is 4.
    activation : str, optional
        Activation function for intermediate layers. Default is 'relu'.
    name : str, optional
        Model name. Default is 'DepthwiseSplitOutputTransPhaseNet'.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the model.
    model : keras.Model
        Full Keras model for inference and training.
    """
    def __init__(
        self,
        num_channels,
        filters=None,
        kernelsizes=None,
        apply_conv_before_block=True,
        output_activation='sigmoid',
        kernel_regularizer=None,
        dropout_rate=0.2,
        initializer='glorot_normal',
        residual_attention=None,
        pool_type='max',
        att_type='across',
        num_transformers=1,
        num_heads=4,
        rnn_type='lstm',
        additive_att=True,
        stacked_layer=4,
        activation='relu',
        name='DepthwiseSplitOutputTransPhaseNet'
    ):
        super(DepthwiseSplitOutputTransPhaseNet, self).__init__(
            num_channels=num_channels,
            num_classes=None,  # let child manage final heads
            filters=filters,
            kernelsizes=kernelsizes,
            apply_conv_before_block=apply_conv_before_block,
            output_activation=output_activation,
            kernel_regularizer=kernel_regularizer,
            dropout_rate=dropout_rate,
            pool_type=pool_type,
            activation=activation,
            initializer=initializer,
            residual_attention=residual_attention,
            att_type=att_type,
            rnn_type=rnn_type,
            additive_att=additive_att,
            stacked_layer=stacked_layer,
            num_transformers=num_transformers,
            num_heads=num_heads,
            name=name,
        )
        # Our child won't rely on parent's final decode
        # We'll manually build 2 separate decoders for P and S

    def build_encoder(self, inputs):
        """
        Build the encoder part of the network.
        Returns the bottleneck features and skip connections.
        """
        # Entry block with depthwise convolution
        x = DepthwiseResnetBlock1D(self.filters[0], 
                          (self.filters[0] // self.num_channels),
                          self.kernelsizes[0], 
                          activation=self.activation, 
                          dropout=self.dropout_rate)(inputs)

        # Collect skip-connections
        skips = [x]

        # Down blocks (1..N-1) using depthwise convolutions
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, skips[-1], self.residual_attention[i])
                if self.additive_att:
                    x += att
                else:
                    x = crop_and_add(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
            skips.append(x)

        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)

        return x, skips

    def build_decoder(self, x, skips, input_shape, name_suffix):
        """
        Build a decoder path from encoded features and skip connections.
        Returns the final output tensor for either P or S head.
        """
        # Up blocks using regular convolutions (not depthwise for decoder)
        for i in range(1, len(self.filters)):
            x = self._up_block(
                self.filters[::-1][i],
                self.kernelsizes[::-1][i],
                x
            )
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)

        # Fix any shape mismatch with the input
        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)

        # Final layers
        x = tfl.Conv1D(
            self.filters[0],
            self.kernelsizes[0],
            strides=1,
            kernel_regularizer=self.kernel_regularizer,
            padding="same",
            name=f'exit_{name_suffix}'
        )(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Single-channel output with sigmoid activation
        x = tfl.Conv1D(1, 1, padding="same")(x)
        return tfl.Activation('sigmoid', dtype='float32', name=name_suffix)(x)

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        # Build encoder
        x, skips = self.build_encoder(inputs)

        # Save the encoder model if needed
        self.encoder = tf.keras.Model(inputs, x, name="encoder")

        # Build P and S decoders
        p_out = self.build_decoder(x, skips, input_shape, name_suffix="p_head")
        s_out = self.build_decoder(x, skips, input_shape, name_suffix="s_head")

        # Build the final model with two outputs
        self.model = tf.keras.Model(inputs=inputs, outputs=[p_out, s_out], name=self.name)

    def call(self, inputs):
        return self.model(inputs)

# -----------------------------------------------------------------------------
#  SplitOutputBranchDepthwise: Depthwise + Branch variant
# -----------------------------------------------------------------------------
class SplitOutputBranchDepthwise(DepthwiseTransPhaseNet):
    """
    SplitOutputBranchDepthwise: Efficient Branch Model with Depthwise Convolutions.

    Combines the efficiency of depthwise separable convolutions with the memory/speed
    trade-off of branch-after-N decoder sharing for P and S phase detection.

    Parameters
    ----------
    num_channels : int
        Number of input channels.
    branch_at : int, optional
        Index at which to branch the decoder. If None, shares the entire decoder.
        branch_at = 0      → identical to the original dual-decoder (slow)
        branch_at = N      → share **all** up-blocks, only 1×1 heads differ
         where *N* = ``len(filters)``.  The public output interface stays the same:
        two tensors ``[p_out, s_out]`` of shape *(batch, time, 1)* with *sigmoid*
        activations so existing loss/metric helpers work unchanged.
    filters : list of int, optional
        List of filter counts for each block. Default is None.
    kernelsizes : list of int, optional
        List of kernel sizes for each block. Default is None.
    apply_conv_before_block : bool, optional
        Whether to apply a Conv1D before depthwise blocks. Default is True.
    output_activation : str, optional
        Activation function for the output layer. Default is 'sigmoid'.
    kernel_regularizer : keras.regularizers.Regularizer, optional
        Kernel regularizer for convolutional layers. Default is None.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    initializer : str or keras.initializers.Initializer, optional
        Weight initializer. Default is 'glorot_normal'.
    residual_attention : list of int, optional
        List of residual attention sizes. Default is [16, 16, 16, 16].
    pool_type : {'max', 'avg'}, optional
        Type of pooling layer. Default is 'max'.
    att_type : {'downstep', 'across'}, optional
        Attention type. Default is 'across'.
    rnn_type : {'lstm', 'causal', 'depthwise', 'none'}, optional
        Type of context modeling. Default is 'lstm'.
    additive_att : bool, optional
        If True, use additive attention. Default is True.
    stacked_layer : int, optional
        Number of stacked residual layers for causal context. Default is 4.
    activation : str, optional
        Activation function for intermediate layers. Default is 'relu'.
    name : str, optional
        Model name. Default is 'SplitOutputBranchDepthwise'.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the model.
    model : keras.Model
        Full Keras model for inference and training.
    branch_at : int
        Branching index for decoder splitting.
    """
    def __init__(self, num_channels, branch_at: int = None, *args, **kwargs):
        # Drop any user-supplied ``num_classes`` – handled internally
        kwargs.pop("num_classes", None)
        super().__init__(num_channels=num_channels, num_classes=None, *args, **kwargs)

        total_blocks = len(self.filters) - 1  # number of up-sampling steps
        # Default: share the **whole** decoder if user leaves the arg empty.
        self.branch_at = total_blocks if branch_at is None else int(branch_at)
        if self.branch_at < 0 or self.branch_at > total_blocks:
            raise ValueError(
                f"branch_at must be in [0, {total_blocks}], got {branch_at}")

    def _build_encoder(self, inputs):
        """Return bottleneck tensor *x* and list of skip tensors."""
        # Entry block with depthwise convolution
        x = DepthwiseResnetBlock1D(
            self.filters[0], 
            (self.filters[0] // self.num_channels),
            self.kernelsizes[0], 
            activation=self.activation, 
            dropout=self.dropout_rate
        )(inputs)

        # Collect skip-connections
        skips = [x]

        # Down blocks (1..N-1) using depthwise convolutions
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, skips[-1], self.residual_attention[i])
                if self.additive_att:
                    x += att
                else:
                    x = crop_and_add(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
            skips.append(x)

        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)

        return x, skips

    def _build_shared_decoder(self, x, skips, input_shape, up_to: int):
        """Build shared decoder up to the specified block index."""
        for i in range(1, up_to + 1):
            x = self._up_block(
                self.filters[::-1][i],
                self.kernelsizes[::-1][i],
                x
            )
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)
        return x

    def _build_task_head(self, x, skips, input_shape, name_suffix):
        """Build task-specific head for P or S detection."""
        # Complete the remaining decoder blocks
        total_blocks = len(self.filters) - 1
        for i in range(self.branch_at + 1, total_blocks + 1):
            x = self._up_block(
                self.filters[::-1][i],
                self.kernelsizes[::-1][i],
                x
            )
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[::-1][i], 1, padding='same')(x)

        # Fix any shape mismatch with the input
        to_crop = x.shape[1] - input_shape[1]
        if to_crop != 0:
            of_start, of_end = to_crop // 2, to_crop // 2
            of_end += to_crop % 2
            x = tfl.Cropping1D((of_start, of_end))(x)

        # Final layers
        x = tfl.Conv1D(
            self.filters[0],
            self.kernelsizes[0],
            strides=1,
            kernel_regularizer=self.kernel_regularizer,
            padding="same",
            name=f'exit_{name_suffix}'
        )(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        # Single-channel output with sigmoid activation
        x = tfl.Conv1D(1, 1, padding="same")(x)
        return tfl.Activation('sigmoid', dtype='float32', name=name_suffix)(x)

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape[1:])

        # Build encoder
        x, skips = self._build_encoder(inputs)

        # Save the encoder model if needed
        self.encoder = tf.keras.Model(inputs, x, name="encoder")

        # Build shared decoder up to branch_at
        shared_x = self._build_shared_decoder(x, skips, input_shape, self.branch_at)

        # Build task-specific heads
        p_out = self._build_task_head(shared_x, skips, input_shape, name_suffix="p_head")
        s_out = self._build_task_head(shared_x, skips, input_shape, name_suffix="s_head")

        # Build the final model with two outputs
        self.model = tf.keras.Model(inputs=inputs, outputs=[p_out, s_out], name=self.name)

    def call(self, inputs):
        return self.model(inputs)

# -----------------------------------------------------------------------------
#  SplitOutputTransPhaseNet variant with *branch-after-N* decoder sharing
# -----------------------------------------------------------------------------
class SplitOutputTransPhaseNetBranch(TransPhaseNet):
    """
    SplitOutputTransPhaseNetBranch: Memory/Speed Trade-off Variant of SplitOutputTransPhaseNet.

    Shares the decoder for the first N up-sampling blocks, then splits into two task-specific tails for P and S phase outputs.
    Allows for a trade-off between memory usage and speed by adjusting the branching point.

    Parameters
    ----------
    branch_at : int, optional
        Index at which to branch the decoder. If None, shares the entire decoder.
        branch_at = 0      → identical to the original dual-decoder (slow)
        branch_at = N      → share **all** up-blocks, only 1×1 heads differ
         where *N* = ``len(filters)``.  The public output interface stays the same:
        two tensors ``[p_out, s_out]`` of shape *(batch, time, 1)* with *sigmoid*
        activations so existing loss/metric helpers work unchanged.
        
    *args, **kwargs :
        Additional arguments passed to the parent class.

    Attributes
    ----------
    encoder : keras.Model
        Encoder part of the model.
    model : keras.Model
        Full Keras model for inference and training.
    branch_at : int
        Branching index for decoder splitting.
    """
    def __init__(self, *args, branch_at: int = None, **kwargs):
        # Drop any user-supplied ``num_classes`` – handled internally
        kwargs.pop("num_classes", None)
        super().__init__(num_classes=None, *args, **kwargs)

        total_blocks = len(self.filters) - 1  # number of up-sampling steps
        # Default: share the **whole** decoder if user leaves the arg empty.
        self.branch_at = total_blocks if branch_at is None else int(branch_at)
        if self.branch_at < 0 or self.branch_at > total_blocks:
            raise ValueError(
                f"branch_at must be in [0, {total_blocks}], got {branch_at}")

    # ------------------------------------------------------------------
    #  Encoder (reuse implementation from SplitOutputTransPhaseNet)
    # ------------------------------------------------------------------
    def _build_encoder(self, inputs):
        """Return bottleneck tensor *x* and list of skip tensors."""
        x = ResnetBlock1D(
            self.filters[0], self.kernelsizes[0],
            activation=self.activation, dropout=self.dropout_rate,
        )(inputs)
        skips = [x]
        for i in range(1, len(self.filters)):
            x = self._down_block(self.filters[i], self.kernelsizes[i], x)
            if self.residual_attention[i] > 0 and self.att_type == 'downstep':
                att = self._att_block(x, skips[-1], self.residual_attention[i])
                if self.additive_att:
                    x = crop_and_add(x, att)
                else:
                    x = crop_and_concat(x, att)
                    x = tfl.Conv1D(self.filters[i], 1, padding='same')(x)
            skips.append(x)

        # Optional attention in bottleneck
        if self.residual_attention[-1] > 0:
            att = self._att_block(x, x, self.residual_attention[-1])
            if self.additive_att:
                x = crop_and_add(x, att)
            else:
                x = crop_and_concat(x, att)
                x = tfl.Conv1D(self.filters[-1], 1, padding='same')(x)
        return x, skips
    # ------------------------------------------------------------------
    #  Utility: final alignment + sigmoid head
    # ------------------------------------------------------------------
    def _final_align(self, x, input_shape, name: str):
        to_crop = x.shape[1] - input_shape[1]
        if to_crop:
            s, e = to_crop // 2, to_crop - to_crop // 2
            x = tfl.Cropping1D((s, e))(x)

        x = tfl.Conv1D(
            self.filters[0], self.kernelsizes[0], padding='same')(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.Activation(self.activation)(x)
        x = tfl.Dropout(self.dropout_rate)(x)

        x = tfl.Conv1D(1, 1, padding='same', name=f'exit_{name}')(x)
        return tfl.Activation('sigmoid', dtype='float32', name=name)(x)

    # ------------------------------------------------------------------
    #  Keras build
    # ------------------------------------------------------------------
    def build(self, input_shape):  # noqa: D401
        inputs = tf.keras.Input(shape=input_shape[1:])

        # ----------------- encoder -----------------
        bottleneck, skips = self._build_encoder(inputs)

        # ----------------- shared part -------------
        shared_x = bottleneck
        for i in range(1, self.branch_at):
            f = self.filters[::-1][i]
            kz = self.kernelsizes[::-1][i]
            shared_x = self._up_block(f, kz, shared_x)
            if self.residual_attention[::-1][i] > 0 and self.att_type == 'across':
                att = self._att_block(
                    skips[::-1][i], skips[::-1][i], self.residual_attention[::-1][i],
                )
                if self.additive_att:
                    shared_x = crop_and_add(shared_x, att)
                else:
                    shared_x = crop_and_concat(shared_x, att)
                    shared_x = tfl.Conv1D(f, 1, padding='same')(shared_x)

        # ----------------- tails -------------------
        def _tail(start_x, start_idx: int, name: str):
            x_t = start_x
            for j in range(max(start_idx, 1), len(self.filters)):
                f = self.filters[::-1][j]
                kz = self.kernelsizes[::-1][j]
                x_t = self._up_block(f, kz, x_t)
                if self.residual_attention[::-1][j] > 0 and self.att_type == 'across':
                    att = self._att_block(
                        skips[::-1][j], skips[::-1][j], self.residual_attention[::-1][j],
                    )
                    if self.additive_att:
                        x_t = crop_and_add(x_t, att)
                    else:
                        x_t = crop_and_concat(x_t, att)
                        x_t = tfl.Conv1D(f, 1, padding='same')(x_t)
            return self._final_align(x_t, input_shape, name)

        p_out = _tail(shared_x, self.branch_at, 'p_head')
        s_out = _tail(shared_x, self.branch_at, 's_head')

        self.model = tf.keras.Model(inputs, [p_out, s_out], name=self.name)

    # ------------------------------------------------------------------
    #  Forward wrapper
    # ------------------------------------------------------------------

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


