from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import SegmentationInferenceModel, upsample

variants = ('mobilenet_v2', 'xception_65')


class DeeplabInferenceModel(SegmentationInferenceModel):
    def __init__(self, model_variant, n_classes=21, learned_upsample=False,
                 depth_multiplier=1.0):
        if model_variant not in variants:
            raise ValueError('Invalid model_variant "%s." Must be in %s'
                             % (model_variant, str(variants)))
        self._n_classes = n_classes
        self._learned_upsample = False
        self._model_variant = model_variant
        self._depth_multiplier = depth_multiplier

    def get_inference(self, features, mode):
        from deeplab.core.feature_extractor import extract_features
        image = features['image']
        image = image / 127.5
        features, _ = extract_features(
            image, model_variant=self._model_variant, preprocess_images=False,
            num_classes=self._n_classes, is_training=mode == 'train',
            depth_multiplier=self._depth_multiplier)
        if features.shape[-1].value != self._n_classes:
            features = tf.layers.conv2d(features, self._n_classes, 1)
        logits = upsample(features, 8, self._learned_upsample)
        return logits

    def get_warm_start_settings(self):
        return None

    @property
    def implementation(self):
        return 'deeplab'
