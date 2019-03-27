import torch.nn as nn
from utils.conv_utils import calculate_conv_output_dim, calculate_pool_output_dim
from torch.nn.utils import spectral_norm

# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, 
                 embed_channels = 100):        
        super(Encoder, self).__init__()
        print("Encoder = encoder_v06_UpsampleConv_Linear_3D.py")
        

        conv_net = nn.Sequential()
        conv_linear_net = nn.Sequential()
        
        # Layer 1
        conv_net.add_module("Conv_1", 
                    spectral_norm(nn.Conv3d(1, 
                              16,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_1", 
                        nn.LeakyReLU(0.01, inplace = True))
        
        # Layer 2
        conv_net.add_module("Conv_2a", 
                    spectral_norm(nn.Conv3d(16, 
                              32,
                              kernel_size = 4,
                              stride = 2,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_2a", 
                        nn.LeakyReLU(0.01, inplace = True))
        conv_net.add_module("Conv_2b", 
                    spectral_norm(nn.Conv3d(32, 
                              32,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_2b", 
                        nn.LeakyReLU(0.01, inplace = True))    
        
        # Layer 3
        conv_net.add_module("Conv_3a", 
                    spectral_norm(nn.Conv3d(32, 
                              64,
                              kernel_size = 4,
                              stride = 2,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_3a", 
                        nn.LeakyReLU(0.01, inplace = True))
        conv_net.add_module("Conv_3b", 
                    spectral_norm(nn.Conv3d(64, 
                              64,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_3b", 
                        nn.LeakyReLU(0.01, inplace = True))    
        
        # Layer 4
        conv_net.add_module("Conv_4a", 
                    spectral_norm(nn.Conv3d(64, 
                              128,
                              kernel_size = 4,
                              stride = 2,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_4a", 
                        nn.LeakyReLU(0.01, inplace = True))
        conv_net.add_module("Conv_4b", 
                    spectral_norm(nn.Conv3d(128, 
                              128,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_4b", 
                        nn.LeakyReLU(0.01, inplace = True))    
        
        # Layer 5
        conv_net.add_module("Conv_5a", 
                    spectral_norm(nn.Conv3d(128, 
                              256,
                              kernel_size = 4,
                              stride = 2,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_5a", 
                        nn.LeakyReLU(0.01, inplace = True))
        conv_net.add_module("Conv_5b", 
                    spectral_norm(nn.Conv3d(256, 
                              256,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_5b", 
                        nn.LeakyReLU(0.01, inplace = True))   
        
        # Layer 6
        conv_net.add_module("Conv_6a", 
                    spectral_norm(nn.Conv3d(256, 
                              512,
                              kernel_size = 4,
                              stride = 2,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_6a", 
                        nn.LeakyReLU(0.01, inplace = True))
        conv_net.add_module("Conv_6b", 
                    spectral_norm(nn.Conv3d(512, 
                              512,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_6b", 
                        nn.LeakyReLU(0.01, inplace = True))   

        # Layer 7
        conv_net.add_module("Conv_7a", 
                    spectral_norm(nn.Conv3d(512, 
                              1024,
                              kernel_size = 4,
                              stride = 2,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_7a", 
                        nn.LeakyReLU(0.01, inplace = True))
        conv_net.add_module("Conv_7b", 
                    spectral_norm(nn.Conv3d(1024, 
                              1024,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1, 
                              bias = True)))
        conv_net.add_module("leakyrelu_7b", 
                        nn.LeakyReLU(0.01, inplace = True)) 
        
        conv_linear_net.add_module("Linear_0", 
                    nn.Linear(in_features = 1*1*1*1024,               # in the paper it is 4*4*512
                              out_features = embed_channels,        # s = 16 in the paper
                              bias = True))
        
        
        self.conv_net = conv_net
        self.conv_linear_net = conv_linear_net

        


    def forward(self, input):
        
 
        out = self.conv_net(input)
#         print("Encoder out shape = " + str(out.shape))
        
        
        out = out.view(-1, 1*1*1*1024)
#         print("transform out shape = " + str(out.shape))
        
        out = self.conv_linear_net(out)


        return out


"""
Adding a 3D convolutional module
"""
def add_conv_module(conv_net,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    batch_norm,
                    conv_bias,
                    activation,
                    leakyrelu_const,
                    layer,
                    embed_cube_edge):
    
#     print("Channel in = " + str(channels))
#     print("Layer in = " + str(layer))
    
    # Convolution Module
    conv_net.add_module("Conv_{0}".format(layer), 
                    nn.Conv2d(in_channels, 
                              out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding, 
                              bias = conv_bias))
    
    # Batch Norm Module
    if batch_norm == True:
        conv_net.add_module("BatchNorm_{0}".format(layer), 
#                         nn.BatchNorm3d(channels * ch_mult))
                            nn.BatchNorm2d(out_channels)) 
    
    # Activation Module
    if activation == "leakyrelu":
        conv_net.add_module("leakyrelu_{0}".format(layer), 
                        nn.LeakyReLU(leakyrelu_const, inplace = True))
    elif activation == "relu":
        conv_net.add_module("relu_{0}".format(layer), 
                        nn.ReLU(inplace = True)) 
    elif activation == "tanh":
        conv_net.add_module("tanh_{0}".format(layer), 
                        nn.Tanh())
    elif activation == "sigmoid":
        conv_net.add_module("sigmoid_{0}".format(layer), 
                            nn.Sigmoid()) 
        
        
    embed_cube_edge = calculate_conv_output_dim(D = embed_cube_edge,
                                                       K = kernel_size,
                                                       P = padding,
                                                       S = stride)
#     channels = channels * ch_mult
    channels = out_channels
    layer = layer + 1
#     print("Channel out = " + str(channels))
#     print("Layer out = " + str(layer))
    print("Cube Edge out = " + str(embed_cube_edge))
    
    return conv_net, channels, layer, embed_cube_edge
    
    