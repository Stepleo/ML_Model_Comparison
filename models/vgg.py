import torch
import torch.nn as nn
from .layers import conv_block

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
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2), # Only two classes as we're gonna be doing binary classification
        )

    def encoder(self, inputs):
        x1 = self.conv_block_1(inputs)
        p1 = self.pool(x1)
        x2 = self.conv_block_2(p1)
        p2 = self.pool(x2)
        x3 = self.conv_block_3(p2)
        p3 = self.pool(x3)
        x4 = self.conv_block_4(p3)
        p4 = self.pool(x4)
        x5 = self.conv_block_5(p4)
        features = self.pool(x5)
        return features

    def classification(self, features):
        x_avg = self.avgpool(features)
        x_avg_flat = torch.flatten(x_avg, 1)
        x = self.classifier(x_avg_flat)
        return x

    def forward(self, inputs):
        # Features extraction
        features = self.encoder(inputs)
        # Classification
        x = self.classification(features)

        return x