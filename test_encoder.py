import themo.data as data
import themo.embedder as embedder

dset = data.WITParallel(datadir="/home/ouhenio/storage/datasets/themo", split="val", download=True)
clipEmbedder = embedder.FrozenCLIPEmbedder()

print(dset[0])