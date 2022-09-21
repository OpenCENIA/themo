[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# <p align="center">Themo</p>

Themo, named after [Themo Lobos](https://es.wikipedia.org/wiki/Themo_Lobos), is a BERT-based CLIP text encoder trained in spanish.

## Training

To train your own version of Themo, simply run
```console
python -m themo train
```

## Evaluation

Our best results were achieved with the following hyperparameters
```console
python -m themo train --batch-size 256 --learn-rate 8e-5
```

Which achieved a final training loss of 0.244 and the following evaluation scores

|           |  @01  |  @05  |  @10  |
|-----------|:-----:|:-----:|:-----:|
| Accuracy  | 0.366 | 0.586 | 0.649 |
| Retrieval | 0.481 | 0.752 |  0.85 |

To evaluate your trained model, run (something like)

```console
python -m themo test --version-path logs/.../version_X
```

For the sake of comparisson, here are the baseline results (taken from [Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP))
|           |  @01  |  @05  |  @10  |
|-----------|:-----:|:-----:|:-----:|
| Accuracy  | 0.370 | 0.594 | 0.660 |
| Retrieval | 0.504 | 0.795 | 0.888 |

They may also be accessed running

```
python -m themo test --baseline
```

## Data

Some data is kinda tricky to get and/or is super redundant bcause we use e.g
only the test set, etc. so for simplicity here are some instructions on how to
download the data we are using.

### MSCOCO / XTD10
The captions come from the official repo of XTD10 and the implementation takes care of
downloading. The images come from standard MSCOCO, but not all images are used. To
download the filtered version run
```console
mkdir -p data/mscoco && wget -O- https://users.dcc.uchile.cl/\~gchapero/datasets/coco_xtd10.tar.gz | tar -xz -C data/mscoco
```

You can use full MSCOCO but it is disk-inefficient. Anyways, the data dir should
look like this for the images to be located properly

```
data
...
├── mscoco
│   ├── train2014
│   │   ...
│   │   ├── COCO_train2014_000000436508.jpg
│   │   ├── COCO_train2014_000000436515.jpg
│   │   ...
│   └── val2014
│       ...
│       ├── COCO_val2014_000000127068.jpg
│       ├── COCO_val2014_000000127074.jpg
│       ...
...
```

The command I gave should leave things in this format. Any extra dirs and files are ignored, so you can use full mscoco if you want.


### ImageNet

Same as with MSCOCO, you can use the full ImageNet in the datadir, but the train images aren't needed. The following command only downloads the splits needed for this work.
```console
mkdir -p data/imagenet && wget -O- https://users.dcc.uchile.cl/\~gchapero/datasets/imagenet_object_localization_patched2019_val_test_only.tar.gz | tar -xzC data/imagenet
```

The datadir should end up looking like this, whether you use full ImageNet or our filtered version

```
data/
├── imagenet
│   ├── ILSVRC
│   │   ├── Annotations
│   │   │   └── CLS-LOC
│   │   │       └── val
│   │   │           ├── ILSVRC2012_val_00000001.xml
│   │   │           ├── ILSVRC2012_val_00000002.xml
│   │   │           └── ...
│   │   └── Data
│   │       └── CLS-LOC
│   │           ├── test
│   │           │   ├── ILSVRC2012_test_00000001.JPEG
│   │           │   ├── ILSVRC2012_test_00000002.JPEG
│   │           │   └── ...
│   │           └── val
│   │               ├── ILSVRC2012_val_00000001.JPEG
│   │               ├── ILSVRC2012_val_00000002.JPEG
│   │               └── ...
│   ├── LOC_sample_submission.csv
│   ├── LOC_synset_mapping.txt
│   ├── LOC_train_solution.csv
│   └── LOC_val_solution.csv
└── ...
```

Any extra dirs or files are ignored, so that you can use the full ImageNet if you have it at hand.
