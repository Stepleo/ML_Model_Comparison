import torch
import torch.nn as nn

    
class conv_block(nn.Module):
    """
    Class that creates a convolution block to go from an image with 
    in_c channels to an image with out_c channels in n_conv convolutions.
    Kernel size and stride are set following the VGG paper.
    """
    def __init__(self, in_c, out_c, n_conv):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.convs = self.build_convs(in_c, out_c, n_conv)

    def build_convs(self, in_c, out_c, n_conv):
        convs = []
        for _ in range(n_conv):
            conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            convs += [conv, nn.BatchNorm2d(out_c), nn.ReLU()]
            in_c = out_c # The image following the first conv will have out_c channels
        return nn.Sequential(*convs)         

    def forward(self, inputs):
        # Apply the layers sequentially defined in build_convs
        x = self.convs(inputs)
        return x

class residual_block(nn.Module):
    """
    Adds residual connection to a conv_block.
    """
    def __init__(self, conv_block: conv_block):
        super().__init__()
        self.conv_block = conv_block
        # Downsampling for the residual
        self.downsample = nn.Sequential(
            nn.Conv2d(conv_block.in_c, conv_block.out_c, kernel_size=1, stride=1),
            nn.BatchNorm2d(conv_block.out_c),
        )

    def forward(self, inputs):
        residual = self.downsample(inputs)
        x = self.conv_block(inputs)
        return x + residual
    
class decoder_block(nn.Module):
    """
    Decoder block for the UNet architecture
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # Upsampling by Transpose
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv_skip = conv_block(out_c + out_c, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs, skip = None):
        x = self.up(inputs)
        if skip is not None: # Allow for skip to be none to use the same decoder blocks for VAE
            # Concatenate with skip connection
            x = torch.cat([x, skip], axis=1)
            x = self.conv(x)
        x_norm = self.bn(x)
        x_act = self.relu(x_norm)
        return x_act
    
