import themo.data as data

dset = data.WITParallel(datadir="/home/ouhenio/storage/datasets/themo", split="val", download=True)

print(len(dset[0][2]))
