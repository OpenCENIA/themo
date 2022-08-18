[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# themo

# data

Some data is kinda tricky to get and/or is super redundant bcause we use e.g
only the test set, etc. so for simplicity here are some instructions on how to
download the data we are using.

## COCO / XTD10
The captions come from the official repo of XTD10 and the implementation takes care of
downloading. The images come from standard COCO, but not all images are used. To
download the filtered version run
```console
mkdir -p data/coco && wget -O- https://users.dcc.uchile.cl/\~gchapero/datasets/coco_xtd10.tar.gz | tar -xz
```

## ImageNet
WIP
