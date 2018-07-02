#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags, app
from tf_template.cli import coord_main, register_coord_fn
from voc_vgg.predictions import save_predictions, vis_saved_predictions

flags.DEFINE_string('model_id', default='base', help='id of model')
flags.DEFINE_string(
    'dst_id', default=None,
    help='id of destination model when action=="copy_to"')


def copy_to(coord):
    from voc_vgg.coordinator import get_coordinator, get_params_path
    import os
    import shutil
    dst_id = flags.FLAGS.dst_id
    dst_path = get_params_path(dst_id)
    if not os.path.isfile(dst_path):
        src_path = get_params_path(flags.FLAGS.model_id)
        print('Copying params...')
        shutil.copyfile(src_path, dst_path)

    dst = get_coordinator(dst_id)
    if os.path.isdir(dst.model_dir):
        raise IOError('Directory already exists for dst: %s' % dst.model_dir)
    print('Copying model_dir...')
    shutil.copytree(coord.model_dir, dst.model_dir)
    print('Done!')


def main(_):
    from voc_vgg.coordinator import get_coordinator
    coord_main(get_coordinator(flags.FLAGS.model_id))


register_coord_fn('save_predictions', save_predictions)
register_coord_fn('vis_saved_predictions', vis_saved_predictions)
register_coord_fn('copy_to', copy_to)

app.run(main)
