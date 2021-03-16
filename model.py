#SegNet
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        
        C_in = 1
        init_f= 64
        num_outputs = 1

        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(8*init_f, 16*init_f, kernel_size=3,padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up1 = nn.Conv2d(16*init_f, 8*init_f, kernel_size=3,padding=1)
        self.conv_up2 = nn.Conv2d(8*init_f, 4*init_f, kernel_size=3,padding=1)
        self.conv_up3 = nn.Conv2d(4*init_f, 2*init_f, kernel_size=3,padding=1)
        self.conv_up4 = nn.Conv2d(2*init_f, init_f, kernel_size=3,padding=1)

        self.conv_out = nn.Conv2d(init_f, num_outputs , kernel_size=3,padding=1)    
    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5(x))

        x=self.upsample(x)
        x = F.relu(self.conv_up1(x))

        x=self.upsample(x)
        x = F.relu(self.conv_up2(x))
        
        x=self.upsample(x)
        x = F.relu(self.conv_up3(x))
        
        x=self.upsample(x)
        x = F.relu(self.conv_up4(x))

        x = self.conv_out(x)
        
        return x 

#UNet
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

#Cosntruct a two-convolutional-layer function
def DoubleConv(in_channel, out_channel):
    doubleconv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True)
    )
    return doubleconv

#Construct up-convolution function
def UpConv(in_channel, out_channel):
    upconv = nn.ConvTranspose2d(
        in_channels = in_channel, 
        out_channels = out_channel,
        kernel_size = 2,
        stride = 2,
    )
    return upconv

#Construct a function to copy and croping from feature map of the contracting path
def Cropping(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta: tensor_size - delta, delta: tensor_size - delta]

#Construct a function to resize the output from upsamplings
def Resizing(tensor, target_tensor):
    newtensor = resize(tensor, size = target_tensor.size()[2])
    return newtensor

#Construct the UNet model
class UNet(nn.Module):
    #Define functions in contracting and expansive path
    def __init__(self):
        super(UNet, self).__init__()
        #Downsampling by maxpooling
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Double convolutional layers in contracting path
        self.down_conv_1 = DoubleConv(1, 64)
        self.down_conv_2 = DoubleConv(64,128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)
        self.down_conv_5 = DoubleConv(512, 1024)

        #Upsampling by tranpose convolution('up-convolution')
        self.up_trans_1 = UpConv(1024, 512)
        self.up_trans_2 = UpConv(512, 256)
        self.up_trans_3 = UpConv(256, 128)
        self.up_trans_4 = UpConv(128, 64)

        #Double convolutional layers in expansive path
        self.up_conv_1 = DoubleConv(1024, 512)
        self.up_conv_2 = DoubleConv(512, 256)
        self.up_conv_3 = DoubleConv(256, 128)
        self.up_conv_4 = DoubleConv(128, 64)

        #Output convolutional layer
        self.out = nn.Conv2d(
            in_channels = 64,
            out_channels = 1,
            kernel_size = 1,
        )
    
    #Build up the model
    def forward(self, image):
        #Encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        #Decoder
        x10 = self.up_trans_1(x9)
        x11 = self.up_conv_1(torch.cat([x10, x7], 1))

        x12 = self.up_trans_2(x11)
        x13 = self.up_conv_2(torch.cat([x12, x5], 1))

        x14 = self.up_trans_3(x13)
        x15 = self.up_conv_3(torch.cat([x14, x3], 1))

        x16 = self.up_trans_4(x15)
        x17 = self.up_conv_4(torch.cat([x16, x1], 1))
        
        x_out = self.out(x17)
        return x_out
        

# if __name__ == '__main__':
#     model = UNet()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model=model.to(device)
#     from torchsummary import summary
#     summary(model, input_size=(1, 192, 192))

