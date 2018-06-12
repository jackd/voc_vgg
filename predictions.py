from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def get_predictions_dir(coord):
    return os.path.join(coord.model_dir, 'predictions')


def central_crop(oh, ow, h, w, image):
    hs = (oh - h) // 2
    ws = (ow - w) // 2
    return image[hs: hs + h, ws: ws + w]


def save_predictions(coord):
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    from progress.spinner import Spinner

    def model_fn(features, labels, mode):
        spec = coord.get_estimator_spec(features, labels, mode)
        predictions = spec.predictions
        key = features['key']
        if isinstance(predictions, tf.Tensor):
            predictions = dict(pred=predictions, key=key)
        elif isinstance(predictions, dict):
            predictions = predictions.copy()
            predictions['key'] = key
            predictions['height'] = features['height']
            predictions['width'] = features['width']
        else:
            raise RuntimeError(
                'Unrecognized predictions type: "%s"' % predictions)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    predictions_dir = get_predictions_dir(coord)
    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)
    estimator = tf.estimator.Estimator(model_fn, coord.model_dir)
    print('Generating predictions...')
    spinner = Spinner()
    for prediction in estimator.predict(
            lambda: coord.get_inputs(tf.estimator.ModeKeys.PREDICT)):
        key = prediction['key']
        pred = prediction['pred']
        oh, ow = pred.shape
        pred = central_crop(
            oh, ow, prediction['height'], prediction['width'], pred)
        pred = Image.fromarray(pred.astype(np.uint8))
        pred.save(os.path.join(predictions_dir, '%s.png' % key))
        spinner.next()
    spinner.finish()


def get_saved_predictions(coord):
    from PIL import Image
    from dids.file_io.file_dataset import FileDataset
    predictions_dir = get_predictions_dir(coord)
    if not os.path.isdir(predictions_dir):
        raise RuntimeError(
            'No prediction data at "%s". '
            'Have you run `--action=save_predictions`?')
    dataset = FileDataset(predictions_dir)
    dataset = dataset.map_keys(lambda x: '%s.png' % x, lambda x: x[:-4])
    dataset = dataset.map(lambda fp: Image.open(fp))
    return dataset


def vis_saved_predictions(coord):
    import numpy as np
    import matplotlib.pyplot as plt
    from dids.core import ZippedDataset
    from pascal_voc.dataset import PascalVocDataset
    gt_ds = PascalVocDataset(key='combined', mode='val')
    inf_ds = get_saved_predictions(coord)
    zipped = ZippedDataset(gt_ds, inf_ds)
    with zipped:
        for k, (gt, inf) in zipped.items():
            image = np.array(gt.load_image())
            seg = np.array(gt.load_class_segmentation(), dtype=np.uint8)
            inf = np.array(inf, dtype=np.uint8)
            correct = np.equal(seg, inf).astype(np.uint8)
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
            ax0.imshow(image)
            ax1.imshow(seg)
            ax2.imshow(inf)
            ax3.imshow(correct)
            plt.show()
