#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags, app
from tf_template.cli import coord_main, register_coord_fn
from voc_vgg.predictions import save_predictions, vis_saved_predictions

flags.DEFINE_string('model_id', default='base', help='id of model')


def main(_):
    from voc_vgg.coordinator import get_coordinator
    coord_main(get_coordinator(flags.FLAGS.model_id))


register_coord_fn('save_predictions', save_predictions)
register_coord_fn('vis_saved_predictions', vis_saved_predictions)

app.run(main)
