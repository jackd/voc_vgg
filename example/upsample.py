#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from voc_vgg.inference_model.base import learned_upsample, fixed_upsample
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 21)
y = np.linspace(-1, 1, 21)
X, Y = np.meshgrid(x, y, indexing='ij')
XY = np.stack((X, Y), axis=-1)
c = (np.sum(XY**2, axis=-1) < 0.5).astype(np.float32)

inp = tf.expand_dims(tf.expand_dims(c, 0), -1)
out = tf.squeeze(tf.squeeze(learned_upsample(inp, 2), axis=-1), axis=0)
fout = tf.squeeze(tf.squeeze(fixed_upsample(inp, 2), axis=-1), axis=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    co, fo = sess.run((out, fout))

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(c)
ax1.imshow(co)
ax2.imshow(fo)
plt.show()
# up = learned_upsample(x, 2)
# print(up.shape)
