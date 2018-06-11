Tensorflow implementation of VGG-FCN8s for PASCAL VOC segmentation.

## Setup
1. Get the required gits and add the parent directory to your path.
```
cd /path/to/parent_dir
git clone https://github.com/jackd/tf_template.git  # project structure
git clone https://github.com/jackd/pascal_voc.git   # data
git clone https://github.com/jackd/voc_vgg.git      # implementation
```
2. Add parent directory to your `PYTHONPATH`.
```
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```


## Running
```
cd voc_vgg/scripts/
./vis_inputs.py
./main.py --action=train
tensorboard --logdir=../_models
./main.py --action=eval
./main.py --action=vis_predictions
```

## Customize
To customize, create your own `params` file and modify the existing deserializing functions to accomodate.

## TODO
* slim model.
