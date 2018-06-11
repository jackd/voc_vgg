#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
from tf_template.cli import vis_inputs
FLAGS = flags.FLAGS

flags.DEFINE_integer('height', default=448, help='input height')
flags.DEFINE_integer('width', default=448, help='input height')


def main(_):
    from voc_vgg.data_source import get_data_source
    image_dims = (FLAGS.height, FLAGS.width)

    data_source = get_data_source(image_dims=image_dims)
    vis_inputs(data_source)


app.run(main)
