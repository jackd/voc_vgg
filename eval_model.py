from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_template.eval_model import EvalModel


def get_eval_model(n_classes=21):

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
        accuracy = tf.metrics.accuracy(labels, pred)
        mean_iou = tf.metrics.mean_iou(labels, pred, n_classes)
        mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(
            labels, pred, n_classes)
        return dict(
            accuracy=accuracy,
            mean_iou=mean_iou,
            mean_per_class_accuracy=mean_per_class_accuracy
        )

    return EvalModel(inference_loss, get_eval_metric_ops)
