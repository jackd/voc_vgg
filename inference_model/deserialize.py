from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_inference_model(implementation='keras', **kwargs):
    if implementation == 'keras':
        from .keras import VggKerasInferenceModel
        return VggKerasInferenceModel(**kwargs)
    elif implementation == 'slim':
        from .slim import VggSlimInferenceModel
        return VggSlimInferenceModel(**kwargs)
    elif implementation == 'crf_rnn':
        return get_crfrnn_wrapped_inference_model(**kwargs)
    else:
        raise NotImplementedError(
            'Unrecognized implementation "%s"' % implementation)


def get_crfrnn_wrapped_inference_model(
        base_coord='base', initialize_with_base_variables=True, **kwargs):
    from ..coordinator import get_coordinator
    from .crfrnn_wrapper import CrfRnnInferenceModel
    base_coord = get_coordinator(base_coord)
    base_model = base_coord.inference_model
    if initialize_with_base_variables:
        warmstart_settings = tf.train.latest_checkpoint(base_coord.model_dir)
    return CrfRnnInferenceModel(base_model, warmstart_settings, **kwargs)
