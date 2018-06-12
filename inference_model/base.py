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
    def __init__(self, n_classes=21):
        self.n_classes = n_classes

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
            score2 = tf.layers.conv2d_transpose(
                conv7, n_classes, 4, 2, padding='SAME')

            score_pool4 = tf.layers.conv2d(pool4, n_classes, 1)
            score_fused = score_pool4 + score2
            score4 = tf.layers.conv2d_transpose(
                score_fused, n_classes, 4, 2, padding='SAME', use_bias=False)

            score_pool3 = tf.layers.conv2d(pool3, n_classes, 1)
            score_final = score4 + score_pool3

            upsample = tf.layers.conv2d_transpose(
                score_final, n_classes, 16, 8, padding='SAME', use_bias=False)

        return upsample

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
