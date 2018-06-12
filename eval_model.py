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

        ious = []
        for i in range(n_classes):
            pi = tf.equal(pred, i)
            li = tf.equal(labels, i)
            intersection = tf.logical_and(pi, li)
            union = tf.logical_or(pi, li)
            intersection = tf.cast(intersection, tf.float32)
            union = tf.cast(union, tf.float32)

            intersection = tf.metrics.mean(tf.reduce_sum(intersection))
            union = tf.metrics.mean(tf.reduce_sum(union))

            with tf.control_dependencies([intersection[1], union[1]]):
                iou_val = intersection[0] / union[0]
                iou_op = tf.no_op()
            ious.append((iou_val, iou_op))

        iou_stack = tf.stack(tuple(iou[0] for iou in ious), axis=0)
        mean_iou = tf.reduce_mean(iou_stack)
        with tf.control_dependencies([mean_iou]):
            mean_iou = (mean_iou, tf.no_op())

        return dict(accuracy=accuracy, mean_iou=mean_iou)

    return EvalModel(inference_loss, get_eval_metric_ops)
