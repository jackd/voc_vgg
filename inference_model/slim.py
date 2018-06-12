from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from .base import VggInferenceModel
vgg_16 = nets.vgg.vgg_16


def _download_initial_checkpoint_path(folder='/tmp'):
    import wget
    import tarfile
    path = os.path.join(folder, 'vgg_16_2016_08_28.tar.gz')
    if not os.path.isfile(path):
        url = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'
        path = wget.download(url, path)
    print('\nExtracting...')
    fn = 'vgg_16.ckpt'
    with tarfile.open(path) as tar:
        tar.extractall(folder)
    ckpt_path = os.path.join(folder, fn)
    assert(os.path.isfile(ckpt_path))
    print('Removing archive...')
    os.remove(path)
    print('Done!')
    return ckpt_path


def get_initial_checkpoint_path(folder='/tmp'):
    path = os.path.join(folder, 'vgg_16.ckpt')
    if not os.path.isfile(path):
        _download_initial_checkpoint_path(folder)
    assert(os.path.isfile(path))
    return path


class VggSlimInferenceModel(VggInferenceModel):

    def _get_base_vgg(self, image, mode, load_weights):
        training = mode == 'train'
        logits, endpoints = vgg_16(
            image, spatial_squeeze=False, is_training=training)
        conv6 = endpoints['vgg_16/pool5']
        pool4 = endpoints['vgg_16/pool4']
        pool3 = endpoints['vgg_16/pool3']
        return conv6, pool4, pool3

    def _get_dense_weights(self):
        path = get_initial_checkpoint_path()
        graph = tf.Graph()
        with graph.as_default():
            input_image = tf.placeholder(
                shape=(None, 224, 224, 3), dtype=tf.float32)
            vgg_16(
                input_image, spatial_squeeze=False, is_training=True)
            w1 = graph.get_tensor_by_name('vgg_16/fc6/weights:0')
            b1 = graph.get_tensor_by_name('vgg_16/fc6/biases:0')
            w2 = graph.get_tensor_by_name('vgg_16/fc7/weights:0')
            b2 = graph.get_tensor_by_name('vgg_16/fc7/biases:0')
            with tf.Session() as sess:
                loader = tf.train.Saver(var_list=[w1, b1, w2, b2])
                sess.run(tf.global_variables_initializer())
                loader.restore(sess, path)
                w1, b1, w2, b2 = sess.run((w1, b1, w2, b2))
        return ((w1, b1), (w2, b2))

    def _create_warm_start_settings(self, folder):
        graph = tf.Graph()
        with graph.as_default():
            image = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
            self._get_inference(image, 'train', load_weights=True)
            fcn8s_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_fcn8s')
            vgg_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg')
            vgg_vars = [v for v in vgg_vars if 'fc' not in v.name]
            loader = tf.train.Saver(var_list=vgg_vars)
            saver = tf.train.Saver(var_list=fcn8s_vars + vgg_vars)
        with tf.Session(graph=graph) as sess:
            loader.restore(sess, get_initial_checkpoint_path())
            sess.run([v.initializer for v in fcn8s_vars])
            saver.save(sess, os.path.join(folder, 'model'))
        return tf.train.latest_checkpoint(folder)

    @property
    def implementation(self):
        return 'slim'
