# Fetal_Segmentation_Pytorch
This repo showcases two deep learning methods(SegNet and UNet) with Pytorch to segment fetal images. The work is heavily based on the book [
PyTorch Computer Vision Cookbook-Michael Avendi](https://www.packtpub.com/product/pytorch-computer-vision-cookbook/9781838644833).
## Repository overview
[data/training/](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/tree/main/data/training): Stores all the unsplited training data <br/>
[data/test_set](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/tree/main/data/test_set): Stores all data for prediction use <br/>
[model/weights.pt](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/models/weights.pt): Stores the best weight model generated after training<br/>
[main.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/main.py): The main script imports dataset, trainer, loss functions to run the model <br/>
[dataset.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/dataset.py): Customise a dataset to process the trainig images <br/>
[model.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/model.py): Construct the SegNet and UNet model <br/>
[train.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/train.py): The trainer to run epochs <br/>
[loss_functions.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/loss_functions.py): Define the dice loss + BCElogits loss function <br/>
[predict.py](https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/predict.py): Script to predict unlabeld images <br/>

## Requirements 
- `torch == 1.8.0`
- `torchvision`
- `torchsummary`
- `numpy`
- `scipy`
- `skimage`
- `matplotlib`
- `PIL`

## SegNet
SegNet Model Results<br/>
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/SegNet%20results/Demo_1.png" width="640" height="177"><br/>
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/SegNet%20results/Demo_2.png" width="640" height="177"><br/>
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/SegNet%20results/loss.png" width="320" height="240">
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/SegNet%20results/acc.png" width="320" height="240"><br/>

## UNet
This UNet implementation is rather a vanilla model, there is no BatchNorm, DropOut utilised. If one follow the original paper strictly, there will be a conflict betweent input and output sizes(572 to 388). To avoid label and prediction mismatch in this implementatino, a resize function has been applied after every up-convolution in expansive path and at final output layer.<br/>
Unet Model Results <br/>
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/UNet%20results/demo_1.png" width="640" height="177"><br/>
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/UNet%20results/demo_2.png" width="640" height="177"><br/>
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/UNet%20results/loss.png" width="320" height="240">
<img src="https://github.com/SimonZeng7108/Fetal_Segmentation_Pytorch/blob/main/UNet%20results/acc.png" width="320" height="240"><br/>
