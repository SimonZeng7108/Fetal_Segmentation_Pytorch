import os
import numpy as np
import torch


path2test="./data/test_set/"
path2weights="./models/weights.pt"
h,w=192,192
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

imgsList=[pp for pp in os.listdir(path2test) if "Annotation" not in pp]
print("number of images:", len(imgsList))
rndImg=np.random.choice(imgsList,1)
print(rndImg)

#Load model
from model import SegNet
params_model={
        "input_shape": (1,h,w),
        "initial_filters": 16, 
        "num_outputs": 1,
            }
model = SegNet()
model = model.to(device)

#Evaluate
from PIL import Image
from torchvision.transforms.functional import to_tensor
model.load_state_dict(torch.load(path2weights))
model.eval()

path2img = os.path.join(path2test, rndImg[0])
img = Image.open(path2img)
img=img.resize((w,h))
img_t=to_tensor(img).unsqueeze(0).to(device)

pred=model(img_t)
pred=torch.sigmoid(pred)[0]
mask_pred= (pred[0]>=0.5).to('cpu')

#Plot the graph
#Define a show mask on image function
from torchvision.transforms.functional import to_pil_image
from skimage.segmentation import mark_boundaries
def show_img_mask(img, mask): 
    if torch.is_tensor(img):
        # img = img.to('cpu')
        # mask = mask.to('cpu')
        img=to_pil_image(img)
        mask=to_pil_image(mask)

    img_mask=mark_boundaries(
        np.array(img), 
        np.array(mask),
        outline_color=(0,1,0),
        color=(0,1,0))
    plt.imshow(img_mask)


import matplotlib.pylab as plt
plt.figure()
plt.subplot(1, 3, 1) 
plt.imshow(img, cmap="gray")

plt.subplot(1, 3, 2) 
plt.imshow(mask_pred, cmap="gray")

plt.subplot(1, 3, 3) 
show_img_mask(img, mask_pred)
plt.show()