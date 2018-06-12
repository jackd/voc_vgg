from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base import SegmentationInferenceModel
from crfrnn.ops import unrolled_crf_rnn


class CrfRnnInferenceModel(SegmentationInferenceModel):
    def __init__(self, base_model, warm_start_settings=None, **crf_rnn_kwargs):
        self._base_model = base_model
        self._warm_start_settings = warm_start_settings
        self._crf_rnn_kwargs = crf_rnn_kwargs

    def get_inference(self, features, mode):
        image = features['image']
        logits = self._base_model.get_inference(features, mode)
        logits = unrolled_crf_rnn(logits, image, **self._crf_rnn_kwargs)
        return logits

    def get_warm_start_settings(self):
        if self._warm_start_settings is None:
            return self._base_model.get_warm_start_settings()
        else:
            return self._warm_start_settings
