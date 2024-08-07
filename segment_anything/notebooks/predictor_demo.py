import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import pdb

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
print("Loading image!")
# image_path = "/mnt/msranlp/zliang/data/CLS-LOC/dev_with_labels/n04118776/ILSVRC2012_val_00016799.JPEG"
# image_path = "/mnt/msranlp/zliang/data/CLS-LOC/dev_with_labels/n01440764/ILSVRC2012_val_00017700.JPEG"
image_path = 'notebooks/images/truck.jpg'

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

predictor = SamPredictor(sam)


print("Processing!")
predictor.set_image(image)

input_box = np.array([425, 600, 700, 875])
# input_box = np.array([7, 41, 492, 357])
# input_box = np.array([117, 193, 492, 310])
# input_box = np.array([148, 5, 445, 369])

input_point = np.array([[575, 750]])
input_label = np.array([0])

# pdb.set_trace()

# best_mask, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False,)
best_mask, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=input_box[None, :], multimask_output=False,)
multi_masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=True,)
print("Saving!")
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_mask(best_mask[0], plt.gca())
# show_box(input_box, plt.gca())
# show_points(input_point, input_label, plt.gca())

# plt.axis('off')
# plt.show()

# plt.savefig('prediction.jpg')
# plt.close()

fig, ax = plt.subplots(2, 2, figsize=(20, 20))

ax[0, 0].imshow(image)
show_mask(best_mask[0], ax[0, 0], random_color=True)
show_box(input_box, ax[0, 0])
show_points(input_point, input_label, ax[0, 0])
ax[0, 0].axis('off')

ax[0, 1].imshow(image)
show_mask(multi_masks[0], ax[0, 1], random_color=True)
show_box(input_box, ax[0, 1])
show_points(input_point, input_label, ax[0, 1])
ax[0, 1].axis('off')

ax[1, 0].imshow(image)
show_mask(multi_masks[1], ax[1, 0], random_color=True)
show_box(input_box, ax[1, 0])
show_points(input_point, input_label, ax[1, 0])
ax[1, 0].axis('off')

ax[1, 1].imshow(image)
show_mask(multi_masks[2], ax[1, 1], random_color=True)
show_box(input_box, ax[1, 1])
show_points(input_point, input_label, ax[1, 1])
ax[1, 1].axis('off')

plt.tight_layout()
# plt.show()

plt.savefig('prediction.jpg')
plt.close()