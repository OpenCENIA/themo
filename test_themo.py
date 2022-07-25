import themo.data as data

dset = data.WITParallel(datadir="/home/ouhenio/storage/datasets/themo", split="val", download=True)

print(dset[0])
