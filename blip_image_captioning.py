import time
start_time = time.time()

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# we associate a model with its preprocessors to make it easier for inference.

model, vis_processors, _ = load_model_and_preprocess(
     name="blip_caption", model_type="base_coco", is_eval=True, device=device
 )

# uncomment to use large model

#model, vis_processors, _ = load_model_and_preprocess(
#    name="blip_caption", model_type="large_coco", is_eval=True, device=device
#)

mid_time = time.time()
print("--- {} seconds ---".format(int(time.time() - start_time)))


print(vis_processors.keys())

for i in range(4):
    raw_image = Image.open(fr"C:\Users\razit\sharif\public\projects\image\Dataset\caption\{i}.jpg").convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    print(model.generate({"image": image}))


    # due to the non-determinstic nature of necleus sampling, you may get different captions.
    #print(model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3))


print("--- {} seconds ---".format(int(time.time() - mid_time)))
