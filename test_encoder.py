from email.mime import image
import themo.data as data
import themo.clip_embedders as clip_embedders
from PIL import Image
import requests

dset = data.WITParallel(datadir="/home/ouhenio/storage/datasets/themo", split="val", download=True)

textEmbedder = clip_embedders.FrozenCLIPTextEmbedder().cuda()

z = textEmbedder.encode(dset[0][1])
