from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import TrainModel


def inference_loss(inference, labels):
    import tensorflow as tf
    valid_mask = labels['valid_mask']
    logits = tf.boolean_mask(inference, valid_mask)
    labels = tf.boolean_mask(labels['segmentation'], valid_mask)
    return tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)


def get_train_model(
        batch_size=8, max_steps=150000, optimizer_key='adam',
        learning_rate=1e-4):
    from tf_template.deserialize import deserialize_optimization_op_fn
    return TrainModel.from_fns(
        inference_loss,
        deserialize_optimization_op_fn(
            key=optimizer_key, learning_rate=learning_rate),
        batch_size, max_steps)
