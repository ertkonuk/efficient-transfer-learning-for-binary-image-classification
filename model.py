import torch
from torch import nn
import torch.nn.functional as F

# . . the VGG operations are separate from the rest of the network
class VGGFeatures(nn.Module):
    # . .constructors just saves the vgg model
    def __init__(self, vgg):
        super(VGGFeatures, self).__init__()
        self.vgg = vgg

    # . . pass the input through the VGG network
    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        # . . flatten the output for the logistic regression
        x = x.view(x.size(0), -1)
        
        return x