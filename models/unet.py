import torch.nn as nn
from layers import conv_block, decoder_block
from resnet import ResNet

class UNet(ResNet):
    """
    UNet inheriting from a ResNet encoder.
    """
    def __init__(self):
        super(UNet, self).__init__()
        self.b = conv_block(512, 1024, 1)
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        p, skip_list = self.encoder(inputs)
        b = self.b(p)
        d1 = self.d1(b, skip_list[3])
        d2 = self.d2(d1, skip_list[2])
        d3 = self.d3(d2, skip_list[1])
        d4 = self.d4(d3, skip_list[0])
        outputs = self.outputs(d4)
        return outputs