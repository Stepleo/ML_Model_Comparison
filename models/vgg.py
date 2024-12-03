import torch.nn as nn
from layers import conv_block

class VGG(nn.Module):
    """
    Follows the VGG_19 architecture described in the following list:
    [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    """
    def __init__(self, input_img_c):
        super(VGG, self).__init__()
        self.conv_block_1 = conv_block(in_c=input_img_c, out_c=64, n_conv=2)
        self.conv_block_2 = conv_block(in_c=64, out_c=128, n_conv=2)
        self.conv_block_3 = conv_block(in_c=128, out_c=256, n_conv=4)
        self.conv_block_4 = conv_block(in_c=256, out_c=512, n_conv=4)
        self.conv_block_5 = conv_block(in_c=512, out_c=512, n_conv=4)
        # Max pooling for encoding
        self.pool = nn.MaxPool2d((2, 2))
        

    def forward(self, inputs):
        x1 = self.conv_block_1(inputs)
        p1 = self.pool(x1)
        x2 = self.conv_block_2(p1)
        p2 = self.pool(x2)
        x3 = self.conv_block_3(p2)
        p3 = self.pool(x3)
        x4 = self.conv_block_4(p3)
        p4 = self.pool(x4)
        x5 = self.conv_block_5(p4)
        p5 = self.pool(x5)

        #TODO: Add classifcation layer
        
        return 