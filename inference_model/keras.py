from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from .base import VggInferenceModel


WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/releases/download/'
    'v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/releases/download/'
    'v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def get_weights_path(include_top=True):
    if include_top:
        weights_path = tf.keras.utils.get_file(
          'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models',
          file_hash='64373286793e3c8b2b4e3219cbf3544b')
    else:
        weights_path = tf.keras.utils.get_file(
          'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          file_hash='6d6bbae143d832006294945121d1f1fc')
    return weights_path


class VggKerasInferenceModel(VggInferenceModel):

    def _get_base_vgg(self, image, mode, load_weights):
        model = tf.keras.applications.VGG16(
            include_top=False, input_tensor=image,
            weights='imagenet' if load_weights else None)
        conv6 = model(image)
        graph = tf.get_default_graph()
        pool4 = graph.get_tensor_by_name('block4_pool/MaxPool:0')
        pool3 = graph.get_tensor_by_name('block3_pool/MaxPool:0')
        return conv6, pool4, pool3

    def _get_dense_weights(self):
        import h5py
        weights_path = get_weights_path(include_top=True)
        wbs = []
        with h5py.File(weights_path, 'r') as weights_data:
            for k in ('fc1', 'fc2'):
                w = np.array(weights_data['%s/%s_W_1:0' % (k, k)])
                b = np.array(weights_data['%s/%s_b_1:0' % (k, k)])
                wbs.append((w, b))
        return tuple(wbs)

    def _create_warm_start_settings(self, folder):
        image = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
        self._get_inference(image, 'train', load_weights=True)
        saver = tf.train.Saver()
        sess = tf.keras.backend.get_session()
        fcn8s_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_fcn8s')
        sess.run([v.initializer for v in fcn8s_vars])
        saver.save(sess, os.path.join(folder, 'model'))
        tf.reset_default_graph()
        return tf.train.latest_checkpoint(folder)

    @property
    def implementation(self):
        return 'keras'
