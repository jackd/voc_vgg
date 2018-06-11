from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import TrainModel


def get_train_model(
        batch_size=8, max_steps=150000,
        optimizer_key='adam', learning_rate=1e-6):
    from tf_template.deserialize import deserialize_optimization_op_fn
    return TrainModel(deserialize_optimization_op_fn(
        key=optimizer_key, learning_rate=learning_rate), batch_size, max_steps)
