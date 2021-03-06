
import os
import torch
from model import UNet, SegNet
#Set parameters
path2train="./data/training"        #Path of train image
path2models= "./models/"            #Path to save best weight
h,w= 200,200                        #Input shape
model = SegNet()                    #Options: UNet(), SegNet()
epochs = 150                        #Number of training iterations
lr=3e-4                             #Optimiser learning rate
factor=0.5                          #Schedule learning rate drop rate


#Define data augmentation
from albumentations import (Rotate, HorizontalFlip, VerticalFlip, Compose, Resize)
transform_train = Compose([
    Resize(h,w), 
    Rotate(limit=35, p=0.5),
    HorizontalFlip(p=0.5), 
    VerticalFlip(p=0.5),
])
transform_val = Resize(h,w)

#Load dataset
from dataset import fetal_dataset
fetal_ds1=fetal_dataset(path2train, transform=transform_train)
fetal_ds2=fetal_dataset(path2train, transform=transform_val)

#Split data into train validation
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices=range(len(fetal_ds1))

for train_index, val_index in sss.split(indices):
    pass

train_ds=Subset(fetal_ds1,train_index)
val_ds=Subset(fetal_ds2,val_index)

#Create DataLoader
from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size=8, shuffle=False)
val_dl = DataLoader(val_ds, batch_size=8, shuffle=False) 


#Load model
import torch
from torchsummary import summary
model = model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
summary(model, input_size=(1, h, w))

#Load Loss Function
from loss_functions import loss_func

#Define Optimizer and Scheduler
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
opt = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=factor, patience=20,verbose=1)

#Load trainer
from train import train_val
if not os.path.exists(path2models):
        os.mkdir(path2models)
params_train={
    "num_epochs": epochs,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2models+"weights.pt",
}
model, loss_hist, metric_hist=train_val(model,params_train)

#Plot graph
import matplotlib.pylab as plt
num_epochs=params_train["num_epochs"]
# plot loss progress
plt.figure(1)
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
# plot accuracy progress
plt.figure(2)
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

