import themo.data as data
import themo.embedder as embedder
import torch

dset = data.WITParallel(datadir="/home/ouhenio/storage/datasets/themo", split="val", download=True)
clipEmbedder = embedder.FrozenCLIPEmbedder().cuda()

encoded, pooled_encoded = clipEmbedder.encode(dset[0][1])


print(f"Prompt: '{dset[0][1]}'\n")
print("CLIP embedding:")
print(encoded)
print("\n")
print(f"embedding dimensions: {encoded.shape}")
print(f"EOS embedding dimensions: {pooled_encoded.shape}")

# print(torch.equal(encoded[0][1], encoded[0][0]))
# print(torch.equal(encoded[0][1], encoded[0][-1]))

