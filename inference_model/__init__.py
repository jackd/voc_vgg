from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_inference_model(n_classes=21, implementation='keras'):
    if implementation == 'keras':
        from .keras import VggKerasInferenceModel
        return VggKerasInferenceModel(n_classes)
    elif implementation == 'slim':
        from .slim import VggSlimInferenceModel
        return VggSlimInferenceModel(n_classes)
    else:
        raise NotImplementedError(
            'Unrecognized implementation "%s"' % implementation)
