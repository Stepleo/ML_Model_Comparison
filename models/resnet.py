from .layers import residual_block
from .vgg import VGG

class ResNet(VGG):
    """
    Inherits the VGG architecture and adds residual connections
    """
    def __init__(self, input_img_c=1):
        super(ResNet, self).__init__(input_img_c)
        self.res_conv_block_1 = residual_block(self.conv_block_1)
        self.res_conv_block_2 = residual_block(self.conv_block_2)
        self.res_conv_block_3 = residual_block(self.conv_block_3)
        self.res_conv_block_4 = residual_block(self.conv_block_4)
        self.res_conv_block_5 = residual_block(self.conv_block_5)

    def encoder(self, inputs):
        x1 = self.res_conv_block_1(inputs)
        p1 = self.pool(x1)
        x2 = self.res_conv_block_2(p1)
        p2 = self.pool(x2)
        x3 = self.res_conv_block_3(p2)
        p3 = self.pool(x3)
        x4 = self.res_conv_block_4(p3)
        p4 = self.pool(x4)
        x5 = self.res_conv_block_5(p4)
        p5 = self.pool(x5)
        skip_connections =  [x1, x2, x3, x4, x5]
        
        return p5, skip_connections

    def forward(self, inputs):
        features, _ = self.encoder(inputs)
        x = self.classification(features)
        return x