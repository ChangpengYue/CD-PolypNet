"""
Modified implementation preserving original functionality with enhanced naming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np
import math
# import efb_net.mynn as mynn
import my_functionals.custom_functional as myF
from config import cfg

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL,'BNFUNC')
    normalizationLayer = layer(in_channels)
    return normalizationLayer


def initialize_weights(*models):
   for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class DynamicFeatureFusionLayer(_ConvNd):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            input_channels, output_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self.gate_control = nn.Sequential(
            mynn.Norm2d(input_channels+1),
            nn.Conv2d(input_channels+1, input_channels+1, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels+1, 1, 1),
            mynn.Norm2d(1),
            nn.Sigmoid()
        )

    def forward(self, base_features, condition_features):
        attention_weights = self.gate_control(torch.cat([base_features, condition_features], dim=1))
        modulated_features = base_features * (attention_weights + 1)
        return F.conv2d(modulated_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
  
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class AdaptivePaddingConv2d(nn.Conv2d):
    def forward(self, input):
        return myF.conv2d_same(input, self.weight, self.groups)

class MultiScaleFeatureEnhancer(_ConvNd):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            input_channels, output_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

        self.gate_control = nn.Sequential(
            mynn.Norm2d(input_channels+1),
            nn.Conv2d(input_channels+1, input_channels+1, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels+1, 1, 1),
            mynn.Norm2d(1),
            nn.Sigmoid()
        )

        # Frequency decomposition configuration
        kernel_size = 7
        sigma = 3
        spatial_grid = torch.stack(torch.meshgrid(
            torch.arange(kernel_size).float(),
            torch.arange(kernel_size).float()
        ), dim=-1)

        center = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1/(2*math.pi*variance)) * torch.exp(
            -torch.sum((spatial_grid - center)**2, dim=-1)/(2*variance)
        )
        gaussian_kernel /= gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(input_channels, 1, 1, 1)

        self.lowpass_filter = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            padding=3,
            kernel_size=kernel_size,
            groups=input_channels,
            bias=False
        )
        self.lowpass_filter.weight.data = gaussian_kernel
        self.lowpass_filter.weight.requires_grad = False

        self.feature_integrator = nn.Conv2d(input_channels*2, input_channels, 1)
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1),
            mynn.Norm2d(input_channels),
            nn.Sigmoid()
        )

    def forward(self, base_features, condition_features):
        batch_size, channels, height, width = base_features.size()
        
        # Frequency decomposition
        low_freq_components = self.lowpass_filter(base_features)
        high_freq_components = base_features - low_freq_components
        combined_features = self.feature_integrator(
            torch.cat([high_freq_components, base_features], dim=1)
        )

        attention_weights = self.gate_control(
            torch.cat([base_features, condition_features], dim=1)
        )
        enhanced_features = combined_features * (attention_weights + 1)

        return F.conv2d(enhanced_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

def verification_test():
    import matplotlib.pyplot as plt

    shape_channels = 8
    shape_features = np.random.normal(size=(1, shape_channels, 10, 10))
    texture_features = np.random.normal(size=(1, 1, 10, 10))

    plt.imshow(shape_features[0, 0])
    plt.show()

    shape_tensor = torch.from_numpy(shape_features).float()
    texture_tensor = torch.from_numpy(texture_features).float()

    test_module = DynamicFeatureFusionLayer(shape_channels, shape_channels,
                                            kernel_size=3, stride=1, padding=1)
    output = test_module(shape_tensor, texture_tensor)
    print('Verification complete')

if __name__ == "__main__":
    verification_test()
