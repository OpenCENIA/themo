from email.mime import image
import themo.data as data
import themo.clip_embedders as clip_embedders
from PIL import Image
import requests

dset = data.WITParallel(datadir="/home/ouhenio/storage/datasets/themo", split="val", download=True)

textEmbedder = clip_embedders.FrozenCLIPTextEmbedder().cuda()
visionEmbedder = clip_embedders.FrozenCLIPVisionEmbedder().cuda()

# encoded, pooled_encoded = textEmbedder.encode(dset[0][1])


# print(f"Prompt: '{dset[0][1]}'\n")
# print("CLIP embedding:")
# print(encoded)
# print("\n")
# print(f"embedding dimensions: {encoded.shape}")
# print(f"EOS embedding dimensions: {pooled_encoded.shape}")

test_positive_prompt = "two cats sleeping"
test_negative_prompt = "orange juice"

test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(test_image_url, stream=True).raw)

_, positive_text_token = textEmbedder.encode(test_positive_prompt)
_, negative_text_token = textEmbedder.encode(test_negative_prompt)
_, image_representation = visionEmbedder.encode(image)

print(image_representation)