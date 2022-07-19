import themo.data as data
import themo.embedder as embedder

dset = data.WITParallel(datadir="/home/ouhenio/storage/datasets/themo", split="val", download=True)
clipEmbedder = embedder.FrozenCLIPEmbedder().cuda()

encoded = clipEmbedder.encode(dset[0][1])


print(f"Prompt: '{dset[0][1]}'\n")
print("CLIP embedding:")
print(encoded)
print("\n")
print(f"embedding dimensions: {encoded.shape}")
