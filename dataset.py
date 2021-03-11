
#Creating Custom Dataset
import os
from scipy import ndimage as ndi
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

path2train="./data/training"

class fetal_dataset(Dataset):
    def __init__(self, path2data, transform=None):      

        imgsList=[pp for pp in os.listdir(path2data) if "Annotation" not in pp]
        anntsList=[pp for pp in os.listdir(path2train) if "Annotation" in pp]

        self.path2imgs = [os.path.join(path2data, fn) for fn in imgsList] 
        self.path2annts= [p2i.replace(".png", "_Annotation.png") for p2i in self.path2imgs]

        self.transform = transform
    
    def __len__(self):
        return len(self.path2imgs)

    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        image = Image.open(path2img)

        path2annt = self.path2annts[idx]
        annt_edges = Image.open(path2annt)
        mask = ndi.binary_fill_holes(annt_edges)        
        
        image= np.array(image)
        mask=mask.astype("uint8")        

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']            

        image= to_tensor(image)         
        mask=255*to_tensor(mask)            
        return image, mask

# path2train="./data/training"

# dataset = fetal_dataset(path2train)
# #Create DataLoader
# from torch.utils.data import DataLoader
# train_dl = DataLoader(dataset, batch_size=8, shuffle=False)

# for img_b, mask_b in train_dl:
#     image, mask = img_b[1], mask_b[1]
#     break


# mask_num = mask.squeeze(0)
# mask_num = mask_num.numpy()
# import torch
# from torchvision.transforms.functional import to_pil_image
# from skimage.segmentation import mark_boundaries
# def show_img_mask(img, mask): 
#     if torch.is_tensor(img):
#         img=to_pil_image(img)
#         mask=to_pil_image(mask)

#     img_mask=mark_boundaries(
#         np.array(img), 
#         np.array(mask),
#         outline_color=(0,1,0),
#         color=(0,1,0))
#     plt.imshow(img_mask)

# image=to_pil_image(image)
# mask=to_pil_image(mask)


# import matplotlib.pylab as plt
# plt.figure('demo image')
# plt.subplot(1, 3, 1) 
# plt.imshow(image, cmap="gray")

# plt.subplot(1, 3, 2) 
# plt.imshow(mask, cmap="gray")

# plt.subplot(1, 3, 3) 
# show_img_mask(image, mask)
# plt.show()