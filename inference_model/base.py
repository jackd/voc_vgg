from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tf_template.inference_model import InferenceModel

ws_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_warmstart')


if not os.path.isdir(ws_dir):
    os.makedirs(ws_dir)


def learned_upsample(x, stride):
    kernel_initializer = deconv2d_bilinear_upsampling_initializer
    n_classes = x.shape[-1].value
    return tf.layers.conv2d_transpose(
        x, n_classes, 2*stride, stride, padding='SAME',
        use_bias=False, kernel_initializer=kernel_initializer)


def fixed_upsample(x, stride):
    return tf.image.resize_images(x, tf.shape(x)[1:3] * stride)


def upsample(x, stride, learned):
    return (learned_upsample if learned else fixed_upsample)(x, stride)


def deconv2d_bilinear_upsampling_initializer(
        shape, dtype=None, partition_info=None):
    """
    Initializer for bilinear upsampling via deconvolutions.

    Used in some segmantic segmentation approches such as
    [FCN](https://arxiv.org/abs/1605.06211)

    See `learned_upsample` for example usage.

    Args:
        shape: list of shape
            shape of the filters, [height, width, output_channels, in_channels]

    Returns
        kernel initializer for 2D bilinear upsampling.

    Original version from a commit in
    [tensorlayer](https://github.com/tensorlayer/tensorlayer/blob/
    70a7017ca2473b10a229540f2f64830b8e45aa3c/tensorlayer/layers.py).
    """
    if shape[0] != shape[1]:
        raise Exception(
            'deconv2d_bilinear_upsampling_initializer only supports '
            'symmetrical filter sizes')
    if shape[3] < shape[2]:
        raise Exception(
            'deconv2d_bilinear_upsampling_initializer behaviour is not '
            'defined for num_in_channels < num_out_channels ')
    if dtype is None:
        dtype = tf.float32

    if not dtype.is_floating:
        raise ValueError('dtype must be floating')

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    # Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros(
        [filter_size, filter_size], dtype=dtype.as_numpy_dtype)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    weights = np.zeros(
        (filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    return weights


class SegmentationInferenceModel(InferenceModel):
    def get_predictions(self, features, inference):
        logits = inference
        probs = tf.nn.softmax(logits)
        pred = tf.argmax(logits, axis=-1)
        return dict(pred=pred, probs=probs)

    def prediction_vis(self, prediction_data):
        from tf_template.visualization import VisImage
        return VisImage(prediction_data['pred'])


class VggInferenceModel(SegmentationInferenceModel):
    def __init__(self, n_classes=21, learned_upsample=False):
        self.n_classes = n_classes
        self.learned_upsample = learned_upsample

    def _get_base_vgg(self, input_image, training, load_weights):
        raise NotImplementedError('Abstract method')
        # ...
        # return conv6, pool4, pool3

    def _get_dense_weights(self):
        raise NotImplementedError('Abstract method')
        # ...
        # return (w1, b1), (w2, b2)

    def _create_warm_start_settings(folder):
        raise NotImplementedError('Abstract property')

    @property
    def implementation(self):
        raise NotImplementedError('Abstract property')

    @property
    def _warm_start_folder(self):
        return os.path.join(
            ws_dir, '%s-%d' % (self.implementation, self.n_classes))

    def _get_inference(self, image, mode, load_weights=False):
        training = mode == 'train'
        conv6, pool4, pool3 = self._get_base_vgg(
            image, training, load_weights)
        scope = 'vgg_fcn8s'

        weights = self._get_dense_weights() if load_weights else None
        if weights is None:
            w1, b1, w2, b2 = (None,)*4
        else:
            (w1, b1), (w2, b2) = weights
            w1 = np.reshape(w1, (7, 7, 512, 4096))
            w2 = np.reshape(w2, (1, 1, 4096, 4096))

            def as_fn(x):
                return lambda *args, **kwargs: x

            w1 = as_fn(w1)
            b1 = as_fn(b1)
            w2 = as_fn(w2)
            b2 = as_fn(b2)

        n_classes = self.n_classes

        with tf.variable_scope(scope):
            x = tf.layers.conv2d(
                conv6, 4096, 7, activation=tf.nn.relu, name='conv6',
                kernel_initializer=w1, bias_initializer=b1, padding='SAME')
            x = tf.layers.dropout(x, rate=0.5, training=training)
            x = tf.layers.conv2d(
                x, 4096, 1, activation=tf.nn.relu, name='conv7',
                kernel_initializer=w2, bias_initializer=b2)
            conv7 = tf.layers.dropout(x, rate=0.5, training=training)
            conv7 = tf.layers.conv2d(conv7, n_classes, 1)

            score2 = learned_upsample(conv7, 2)

            score_pool4 = tf.layers.conv2d(pool4, n_classes, 1)
            score_fused = score_pool4 + score2
            score4 = learned_upsample(score_fused, 2)

            score_pool3 = tf.layers.conv2d(pool3, n_classes, 1)
            score_final = score4 + score_pool3
            upsampled_score = upsample(
                score_final, 8, learned=self.learned_upsample)

        return upsampled_score

    def get_inference(self, features, mode):
        logits = self._get_inference(
            features['image'], mode, load_weights=False)
        return logits

    def get_warm_start_settings(self):
        folder = self._warm_start_folder
        path = tf.train.latest_checkpoint(folder)
        if path is None:
            self._create_warm_start_settings(folder)
            path = tf.train.latest_checkpoint(folder)
            assert(path is not None)
        return path
