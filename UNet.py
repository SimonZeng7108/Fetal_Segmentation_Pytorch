import torch
import torch.nn as nn

#Cosntruct a two-convolutional-layer function
def DoubleConv(in_channel, out_channel):
    doubleconv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size = 3),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channel, out_channel, kernel_size = 3),
        nn.ReLU(inplace = True)
    )
    return doubleconv

#Construct up-convolution function
def UpConv(in_channel, out_channel):
    upconv = nn.ConvTranspose2d(
        in_channels = in_channel, 
        out_channels = out_channel,
        kernel_size = 2,
        stride = 2
    )
    return upconv

#Construct function to copy and croping from feature map of the contracting path
def Cropping(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta: tensor_size - delta, delta: tensor_size - delta]

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
            kernel_size = 1
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
        y1 = Cropping(x7, x10)
        x11 = self.up_conv_1(torch.cat([x10, y1], 1))

        x12 = self.up_trans_2(x11)
        y2 = Cropping(x5, x12)
        x13 = self.up_conv_2(torch.cat([x12, y2], 1))

        x14 = self.up_trans_3(x13)
        y3 = Cropping(x3, x14)
        x15 = self.up_conv_3(torch.cat([x14, y3], 1))

        x16 = self.up_trans_4(x15)
        y4 = Cropping(x1, x16)
        x17 = self.up_conv_4(torch.cat([x16, y4], 1))
        
        x_out = self.out(x17)
        return x_out
        
if __name__ == '__main__':
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    from torchsummary import summary
    summary(model, input_size=(1, 572, 572))