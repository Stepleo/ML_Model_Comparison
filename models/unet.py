from resnet import ResNet

class UNet(ResNet):
    def __init__(self):
        super(UNet, self).__init__()
        # Define UNet architecture here

    def forward(self, x):
        # Define forward pass
        return x