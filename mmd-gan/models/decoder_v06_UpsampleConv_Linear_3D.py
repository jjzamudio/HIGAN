import torch.nn as nn
from utils.conv_utils import calculate_deconv_output_dim
import torch

# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, 
                 embed_channels,
                 D_encoder):
        super(Decoder, self).__init__()
        print("Decoder = decoder_v06_UpsampleConv_Linear_3D.py")
        
        
        deconv_linear_net = nn.Sequential()
        deconv_net = nn.Sequential()
    
        deconv_linear_net.add_module("Linear_0", 
                    nn.Linear(in_features = embed_channels, 
                              out_features = 1*1*1*1024, 
                              bias = True))
        
        # Layer 1
        deconv_net.add_module("InterpolateUpsample_1",
                          Interpolate(scale_factor = 2, 
                                      mode = "nearest"))
        deconv_net.add_module("Conv_1",
                          nn.Conv3d(1024, 
                              512,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True))
        deconv_net.add_module("BatchNorm_1", nn.BatchNorm3d(512))
        deconv_net.add_module("relu_1", nn.ReLU(inplace = True)) 
        
        # Layer 2
        deconv_net.add_module("InterpolateUpsample_2",
                          Interpolate(scale_factor = 2, 
                                      mode = "nearest"))
        deconv_net.add_module("Conv_2",
                          nn.Conv3d(512, 
                              256,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True))
        deconv_net.add_module("BatchNorm_2", nn.BatchNorm3d(256))
        deconv_net.add_module("relu_2", nn.ReLU(inplace = True)) 
        
        # Layer 3
        deconv_net.add_module("InterpolateUpsample_3",
                          Interpolate(scale_factor = 2, 
                                      mode = "nearest"))
        deconv_net.add_module("Conv_3",
                          nn.Conv3d(256, 
                              128,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True))
        deconv_net.add_module("BatchNorm_3", nn.BatchNorm3d(128))
        deconv_net.add_module("relu_3", nn.ReLU(inplace = True)) 
        
        # Layer 4
        deconv_net.add_module("InterpolateUpsample_4",
                          Interpolate(scale_factor = 2, 
                                      mode = "nearest"))
        deconv_net.add_module("Conv_4",
                          nn.Conv3d(128, 
                              64,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True))
        deconv_net.add_module("BatchNorm_4", nn.BatchNorm3d(64))
        deconv_net.add_module("relu_4", nn.ReLU(inplace = True)) 
        
        # Layer 5
        deconv_net.add_module("InterpolateUpsample_5",
                          Interpolate(scale_factor = 2, 
                                      mode = "nearest"))
        deconv_net.add_module("Conv_5",
                          nn.Conv3d(64, 
                              32,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True))
        deconv_net.add_module("BatchNorm_5", nn.BatchNorm3d(32))
        deconv_net.add_module("relu_5", nn.ReLU(inplace = True))   
        
        # Layer 6
        deconv_net.add_module("InterpolateUpsample_6",
                          Interpolate(scale_factor = 2, 
                                      mode = "nearest"))
        deconv_net.add_module("Conv_6",
                          nn.Conv3d(32, 
                              16,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True))
        deconv_net.add_module("BatchNorm_6", nn.BatchNorm3d(16))
        deconv_net.add_module("relu_6", nn.ReLU(inplace = True)) 
        
        # Layer 7
        deconv_net.add_module("Conv_7",
                          nn.Conv3d(16, 
                              1,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True))
        deconv_net.add_module("tanh_7", nn.Tanh()) 
#         deconv_net.add_module("sigmoid_5", nn.Sigmoid()) 
        
        self.deconv_linear_net = deconv_linear_net  
        self.deconv_net = deconv_net
        
        


    def forward(self, input):
#         print("\nDecoder - Forward Pass")

        out = self.deconv_linear_net(input)
#         print("deconv_linear out shape = " + str(out.shape))

        out = out.view(-1, 1024, 1, 1, 1)
#         print("transform out shape = " + str(out.shape))
        
        out = self.deconv_net(out)
#         print("deconv out shape = " + str(out.shape))
        
        return out
    

    
    
    
    
    
"""
An Upsample+Convolutional Layer
(Instead of Deconvolutional Layer)
"""
def add_upsampleconv_module(deconv_net,
                    in_channels,
                    out_channels,   # ch_mult,
                    kernel_size,
                    stride,
                    conv_padding,
                    batch_norm,
                    deconv_bias,
                    activation,
                    leakyrelu_const,
                    layer,
                    scale_factor,
                    mode,
                    reflection_padding):
    
#     deconv_net.add_module("UpsampleChecker_{0}".format(layer), 
#                            UpsampleChecker(scale_factor = scale_factor, 
#                                              mode = mode,
#                                              reflection_padding = reflection_padding,
#                                              channels = channels,
#                                              ch_mult = ch_mult,
#                                              kernel_size = kernel_size,
#                                              conv_padding = conv_padding,
#                                              stride = stride,
#                                              deconv_bias = deconv_bias))

    deconv_net.add_module("InterpolateUpsample_{0}".format(layer),
                          Interpolate(scale_factor = scale_factor, 
                                      mode = mode))
                          
    
    deconv_net.add_module("Conv_{0}".format(layer),
                          nn.Conv2d(in_channels, 
                              out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = conv_padding, 
                              bias = deconv_bias))
    
    # Batch Norm Module
    if batch_norm == True:
        deconv_net.add_module("BatchNorm_{0}".format(layer), 
#                         nn.BatchNorm3d(channels // ch_mult)) 
                        nn.BatchNorm2d(out_channels))
    
    # Activation Module
    if activation == "leakyrelu":
        deconv_net.add_module("leakyrelu_{0}".format(layer), 
                        nn.LeakyReLU(leakyrelu_const, inplace = True))
    elif activation == "relu":
        deconv_net.add_module("relu_{0}".format(layer), 
                        nn.ReLU(inplace = True)) 
    elif activation == "tanh":
        deconv_net.add_module("tanh_{0}".format(layer), 
                        nn.Tanh())
    elif activation == "sigmoid":
        deconv_net.add_module("sigmoid_{0}".format(layer), 
                            nn.Sigmoid()) 
        
        
    #channels = channels // ch_mult
    layer = layer + 1
    
    return deconv_net, out_channels, layer    
    
     
    
"""
Adding a 3D Deconvolutional module
"""
def add_deconv_module(deconv_net,
                    in_channels,
                    out_channels, # in_channels * ch_mult
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    deconv_bias,
                    activation,
                    leakyrelu_const,
                    layer):
    
    # Deconvolution Module
    deconv_net.add_module("DeConv_{0}".format(layer), 
                    nn.ConvTranspose2d(in_channels, 
                              out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding, 
                              bias = deconv_bias))
    
    # Batch Norm Module
    if batch_norm == True:
        deconv_net.add_module("BatchNorm_{0}".format(layer), 
                        nn.BatchNorm2d(channels // ch_mult)) 
    
    # Activation Module
    if activation == "leakyrelu":
        deconv_net.add_module("leakyrelu_{0}".format(layer), 
                        nn.LeakyReLU(leakyrelu_const, inplace = True))
    elif activation == "relu":
        deconv_net.add_module("relu_{0}".format(layer), 
                        nn.ReLU(inplace = True)) 
    elif activation == "tanh":
        deconv_net.add_module("tanh_{0}".format(layer), 
                        nn.Tanh())
    elif activation == "sigmoid":
        deconv_net.add_module("sigmoid_{0}".format(layer), 
                            nn.Sigmoid()) 
        
        
    #channels = channels // ch_mult
    layer = layer + 1
    
    return deconv_net, out_channels, layer
    
    
"""
Adding a 3D Upsample-Conv module
This should alleviate the checkerboard problem.
"""
class Interpolate(nn.Module):
    """
    https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    """
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, 
                        scale_factor=self.scale_factor, 
                        mode=self.mode)
        return x   
    

class UpsampleChecker(torch.nn.Module):
    def __init__(self, 
                 scale_factor, 
                 mode,
                 reflection_padding,
                 channels,
                 ch_mult,
                 kernel_size,
                 conv_padding,
                 stride,
                 deconv_bias):
        """
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
          nn.Upsample(scale_factor = 2, mode='bilinear'),
          nn.ReflectionPad2d(1),
          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                             kernel_size=3, stride=1, padding=0)
                             
        nn.Upsample
        mode (string, optional) – the upsampling algorithm: 
        one of nearest, linear, bilinear and trilinear. Default: nearest
        
        For this module to preserve the dimensions of the input cube,
        the kernel = 3, stride = 1 and padding = 0 for the conv
        the padding = 1 for reflection padding
        """
        super(UpsampleChecker, self).__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor, 
                                    mode = mode)
        self.reflection_pad = reflection_padding
        self.conv = nn.Conv2d(channels, 
                              channels // ch_mult,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = conv_padding, 
                              bias = deconv_bias)

    def forward(self, x):
        """
        pad (tuple) – m-elem tuple, where m/2 ≤ input dimensions and m is even
        mode – ‘constant’, ‘reflect’ or ‘replicate’. 
        value – fill value for ‘constant’ padding.
        """
        print("upsample")
        print("out size = " + str(x.size()))
        out = self.upsample(x)
        
#         print("padded")
#         print("out size = " + str(out.size()))
#         out = nn.functional.pad(input = out, 
# #                                 pad = self.reflection_pad, 
#                                 pad = (1,1,1,1,1,1),
#                                 mode='constant', 
#                                 value = 0)
        
        
        print("conv")
        print("out size = " + str(out.size()))
        out = self.conv(out)
        
        print("out size = " + str(out.size()))
        return out


    
    
