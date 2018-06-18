from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import TrainModel


def inference_loss(inference, labels):
    import tensorflow as tf
    valid_mask = labels['valid_mask']
    any_valid = tf.reduce_any(valid_mask)
    logits = tf.boolean_mask(inference, valid_mask)
    labels = tf.boolean_mask(labels['segmentation'], valid_mask)
    return tf.cond(
        any_valid,
        lambda: tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels),
        lambda: tf.zeros(shape=(), dtype=tf.float32)
    )


def add_reg_losses(weight_decay):
    import tensorflow as tf
    # raise Exception()
    if weight_decay is not None and weight_decay > 0:
        reg = tf.contrib.layers.l2_regularizer(weight_decay)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables = [
            v for v in variables if 'weights' in v.name or 'kernel' in v.name]
        for v in variables:
            reg(v)
        # print('--')
        # for v in tf.get_collection(tf.GraphKeys.LOSSES):
        #     print(v)
        # exit()


def get_train_model(
        batch_size=8, max_steps=150000, weight_decay=None,
        optimizer_key='adam', **optimizer_kwargs):
    from tf_template.deserialize import deserialize_optimization_op_fn
    optimizer_kwargs.setdefault('learning_rate', 1e-4)
    return TrainModel.from_fns(
        lambda inf, lab: inference_loss(inf, lab, weight_decay),
        deserialize_optimization_op_fn(
            key=optimizer_key, **optimizer_kwargs),
        batch_size, max_steps,
        before_loss_calc_fn=lambda: add_reg_losses(weight_decay))
