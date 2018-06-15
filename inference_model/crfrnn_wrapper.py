from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .base import SegmentationInferenceModel
from crfrnn.ops import unrolled_crf_rnn


class CrfRnnInferenceModel(SegmentationInferenceModel):
    def __init__(
            self, base_model, warm_start_settings=None, explicit_loop=False,
            num_iterations=10):
        self._base_model = base_model
        self._warm_start_settings = warm_start_settings
        self._fpi_kwargs = dict(
            explicit_loop=explicit_loop, num_iterations=num_iterations)

    def get_inference(self, features, mode):
        image = features['image']
        logits = self._base_model.get_inference(features, mode)
        logits = unrolled_crf_rnn(logits, image, fpi_kwargs=self._fpi_kwargs)
        return logits

    def get_warm_start_settings(self):
        if self._warm_start_settings is None:
            settings = self._base_model.get_warm_start_settings()
            if isinstance(settings, tf.estimator.WarmsStartSettings):
                settings = settings.ckpt_to_initialize_from

            return tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=settings,
                vars_to_warm_start='vgg.*')

        else:
            return self._warm_start_settings
