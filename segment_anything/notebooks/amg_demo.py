import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import pdb

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    
print("Loading image!")
# image_path = "/mnt/msranlp/zliang/data/CLS-LOC/dev_with_labels/n04118776/ILSVRC2012_val_00016799.JPEG"
image_path = "/mnt/msranlp/zliang/data/CLS-LOC/dev_with_labels/n01440764/ILSVRC2012_val_00017700.JPEG"
# image_path = 'notebooks/images/truck.jpg'
# image_path = 'notebooks/images/dog.jpg'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

print("Loading model weights!")
sam_checkpoint = "assets/sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

print("Processing!")

masks = mask_generator.generate(image)

print("Saving!")

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
# plt.axis('off')
# plt.show() 

plt.savefig('prediction.jpg')
plt.close()