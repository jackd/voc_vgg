from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import DataSource
from tf_template.visualization import ImageVis
import numpy as np

IMAGENET_MEAN = (123.68, 116.78, 103.94)


class VocVggDataSource(DataSource):
    def __init__(self, image_dims=(448, 448)):
        self.image_dims = image_dims

    def get_inputs(self, mode, batch_size=None):
        import tensorflow as tf

        from pascal_voc.dataset import PascalVocDataset
        image_dims = self.image_dims
        repeat = mode == 'train'
        shuffle = True
        if mode in {'eval', 'val', 'predict', 'infer', 'test'}:
            mode = 'val'
        elif mode != 'train':
            raise ValueError('Unrecognized mode "%s"' % mode)
        dids_ds = PascalVocDataset('combined', mode=mode)
        dids_ds.open()
        keys = tuple(dids_ds.keys())
        n_keys = len(keys)
        keys = tf.convert_to_tensor(keys, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(keys)

        if repeat:
            dataset = dataset.repeat()

        if shuffle:
            dataset = dataset.shuffle(n_keys)

        if image_dims is None:
            if batch_size is not None and batch_size != 1 or mode == 'train':
                raise ValueError(
                    'image_dims must be defined if batch_size isn\'t 1 or '
                    'mode is train.')

            def map_fn_np(key):
                example = dids_ds[key]
                image = np.array(example.load_image())
                labels = np.array(example.load_class_segmentation())
                return image, labels
        else:
            h_out, w_out = image_dims

            def map_fn_np(key):
                example = dids_ds[key]
                image = np.array(example.load_image())
                labels = np.array(example.load_class_segmentation())

                return image, labels

        def map_fn_tf(key):
            image, labels = tf.py_func(
                map_fn_np, (key,), (tf.uint8, tf.uint8), stateful=False)

            image = tf.cast(image, tf.float32)
            image -= IMAGENET_MEAN

            labels = tf.expand_dims(labels, axis=-1)
            if mode == 'train':
                from deeplab.input_preprocess import preprocess_image_and_label
                # labels = tf.squeeze(labels, axis=-1)
                original_image, image, labels = preprocess_image_and_label(
                    image, labels, *image_dims,
                    min_scale_factor=0.75, max_scale_factor=1.25)

            invalid_mask = tf.equal(labels, 255)
            valid_mask = tf.logical_not(invalid_mask)
            labels = tf.where(invalid_mask, tf.zeros_like(labels), labels)

            if image_dims is not None:
                image.set_shape(image_dims + (3,))
                labels.set_shape(image_dims + (1,))
                valid_mask.set_shape(image_dims + (1,))
                h, w = image_dims
                h = tf.constant(h, dtype=tf.int32)
                w = tf.constant(w, dtype=tf.int32)
            else:
                shape = tf.shape(image)
                h = shape[0]
                w = shape[1]

                d = 512
                image = tf.image.resize_image_with_crop_or_pad(image, d, d)
                labels = tf.image.resize_image_with_crop_or_pad(
                        labels, d, d)
                valid_mask = tf.image.resize_image_with_crop_or_pad(
                        valid_mask, d, d)
                image.set_shape((d, d, 3))
                labels.set_shape((d, d, 1))
                valid_mask.set_shape((d, d, 1))

            labels = tf.cast(labels, tf.int32)
            # image = tf.image.per_image_standardization(image)

            labels = tf.squeeze(labels, axis=-1)
            valid_mask = tf.squeeze(valid_mask, axis=-1)

            return key, image, labels, valid_mask, h, w

        dataset = dataset.map(map_fn_tf, num_parallel_calls=8)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        # dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))
        dataset = dataset.prefetch(1)
        key, image, labels, valid_mask, h, w = \
            dataset.make_one_shot_iterator().get_next()
        features = dict(key=key, image=image, height=h, width=w)
        labels = dict(segmentation=labels, valid_mask=valid_mask)
        return features, labels

    def feature_vis(self, features):
        image = features['image']
        image -= np.min(image)
        image /= np.max(image)
        return ImageVis(image)

    def label_vis(self, label):
        return ImageVis(label['segmentation'])


def get_data_source(**kwargs):
    return VocVggDataSource(**kwargs)
