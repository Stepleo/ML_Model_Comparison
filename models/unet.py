import torch
import torch.nn as nn
from .layers import conv_block, decoder_block
from .resnet import ResNet

class UNet(ResNet):
    """
    UNet inheriting from a ResNet encoder.
    """
    def __init__(self, input_img_size, resnet: ResNet = None):
        super(UNet, self).__init__(input_img_size)
        self.pretrained = resnet is not None
        if self.pretrained:
            self.encoder = resnet.encoder
        self.b = conv_block(512, 1024, 1)
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 512)
        self.d3 = decoder_block(512, 256)
        self.d4 = decoder_block(256, 128)
        self.d5 = decoder_block(128, 64)

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 2)

    def forward(self, inputs):
        # Features extraction
        p, skip_list = self.encoder(inputs)
        b = self.b(p)
        d1 = self.d1(b, skip_list[4])
        d2 = self.d2(d1, skip_list[3])
        d3 = self.d3(d2, skip_list[2])
        d4 = self.d4(d3, skip_list[1])
        d5 = self.d5(d4, skip_list[0])

        # Classification
        x_avg = self.avgpool(d5)
        x_avg_flat = torch.flatten(x_avg, 1)
        x = self.classifier(x_avg_flat)

        return x