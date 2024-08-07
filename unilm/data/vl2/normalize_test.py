from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import torch
import numpy as np
from PIL import Image
import base64
import io

pil_img = Image.open(r"C:\Users\shaohanh\Desktop\tst.png").convert("RGB")
# pt_tensor = np.random.randint(5, size=(224, 224, 3))
# print(pt_tensor)

class NumpyNormalize(torch.nn.Module):
    def __init__(self,  mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor).
        Returns:
        """
        image = np.array(img).transpose(2, 0, 1) # B, H, W, C  -> B, C, H, W
        image = image / 255.0
        image -= np.array(self.mean).reshape(-1, 1, 1)
        image /= np.array(self.std).reshape(-1, 1, 1)
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

tot_func = ToTensor()
norm_func = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

norm_output = norm_func(tot_func(pil_img))

ours_func = NumpyNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
ours_output = ours_func(pil_img)
print((norm_output - ours_output).max().item())

# 2.5775502288105656e-07