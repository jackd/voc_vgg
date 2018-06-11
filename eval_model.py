from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_template.eval_model import EvalModel


def get_eval_model():
    def inference_loss(inference, labels):
        valid_mask = labels['valid_mask']
        logits = tf.boolean_mask(inference, valid_mask)
        labels = tf.boolean_mask(labels['segmentation'], valid_mask)
        return tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)

    def get_eval_metric_ops(predictions, labels):
        valid_mask = labels['valid_mask']
        pred = tf.boolean_mask(predictions['pred'], valid_mask)
        labels = tf.boolean_mask(labels['segmentation'], valid_mask)
        return dict(accuracy=tf.metrics.accuracy(labels, pred))

    return EvalModel(inference_loss, get_eval_metric_ops)
