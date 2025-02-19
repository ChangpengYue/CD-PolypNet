
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch import nn
from efb_net import SEresnext, Resnet
from efb_net.wider_resnet import wider_resnet38_a2
from config import cfg
from efb_net.mynn import initialize_weights, Norm2d
from my_functionals import EFGConv as efg
from utils.loss_mask import JointEdgeSegLoss

EDGE_SAVE_DIR = 'processed_edges'
os.makedirs(EDGE_SAVE_DIR, exist_ok=True)

class TensorCropper(nn.Module):
    def __init__(self, dim, offset):
        super().__init__()
        self.dim = dim
        self.offset = offset

    def forward(self, input_tensor, reference_tensor):
        for axis in range(self.dim, input_tensor.dim()):
            ref_size = reference_tensor.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size, 
                                  device=input_tensor.device)
            input_tensor = input_tensor.index_select(axis, indices)
        return input_tensor

class IdentityMapper(nn.Module):
    def forward(self, input_tensor, reference_tensor):
        return input_tensor

class SideOutputProcessor(nn.Module):
    def __init__(self, channels, kernel_size=None, stride=None, 
                padding=0, enable_crop=True):
        super().__init__()
        self.enable_crop = enable_crop
        self.side_conv = nn.Conv2d(channels, 1, kernel_size=1)
        
        if kernel_size:
            self.upsample_layer = nn.ConvTranspose2d(
                1, 1, kernel_size, stride, padding, bias=False)
            self.crop_layer = TensorCropper(2, kernel_size//4) if enable_crop else IdentityMapper()
        else:
            self.upsample_layer = None

    def forward(self, features, reference=None):
        output = self.side_conv(features)
        if self.upsample_layer:
            output = self.upsample_layer(output)
            output = self.crop_layer(output, reference)
        return output

class ASPPModule(nn.Module):
    def __init__(self, in_channels, reduced_dim=256, output_stride=16, dilation_rates=[6, 12, 18]):
        super().__init__()
        if output_stride == 8:
            dilation_rates = [2*r for r in dilation_rates]
            
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, reduced_dim, 1, bias=False),
                Norm2d(reduced_dim),
                nn.ReLU(inplace=True)
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(in_channels, reduced_dim, 3, 
                         padding=r, dilation=r, bias=False),
                Norm2d(reduced_dim),
                nn.ReLU(inplace=True)
            ) for r in dilation_rates
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_dim, 1, bias=False),
            Norm2d(reduced_dim),
            nn.ReLU(inplace=True)
        )
        
        self.edge_processor = nn.Sequential(
            nn.Conv2d(1, reduced_dim, 1, bias=False),
            Norm2d(reduced_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, edge_input):
        input_size = features.size()
        
        # Global context features
        global_features = self.global_conv(self.global_pool(features))
        global_features = F.interpolate(global_features, input_size[2:], 
                                       mode='bilinear', align_corners=True)
        
        # Edge features processing
        edge_features = self.edge_processor(
            F.interpolate(edge_input, input_size[2:], 
                         mode='bilinear', align_corners=True))
        
        # Feature aggregation
        aspp_output = torch.cat([global_features, edge_features], 1)
        for block in self.aspp_blocks:
            aspp_output = torch.cat([aspp_output, block(features)], 1)
            
        return aspp_output

class EFBranch(nn.Module):
    def __init__(self, num_classes, loss_func=JointEdgeSegLoss(classes=2).cuda(), backbone=None):
        super().__init__()
        self.loss_calculator = loss_func
        self.num_classes = num_classes
        
        # Feature transformation layers
        self.feature_transform_256 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.feature_transform_512 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.feature_transform_4096 = nn.Sequential(
            nn.Conv2d(1024, 4096, 3, padding=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True)
        )
        
        # Edge detection branches
        self.edge_branches = nn.ModuleDict({
            'stage3': SideOutputProcessor(256),
            'stage4': SideOutputProcessor(512),
            'stage7': SideOutputProcessor(4096)
        })
        
        # Multi-scale feature processors
        self.multiscale_processors = nn.Sequential(
            Resnet.BasicBlock(64, 64),
            nn.Conv2d(64, 32, 1),
            Resnet.BasicBlock(32, 32),
            nn.Conv2d(32, 16, 1),
            Resnet.BasicBlock(16, 16),
            nn.Conv2d(16, 8, 1),
            nn.Conv2d(8, 1, 1, bias=False)
        )
        
        # Feature fusion components
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, bias=False)
        )
        
        # Edge feature extractor
        self.edge_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        initialize_weights(self.feature_fusion)

    def _save_edge_detection(self, edge_maps, batch_size):
        for i in range(batch_size):
            edge_img = edge_maps[i].squeeze().cpu().numpy()
            plt.imsave(
                os.path.join(EDGE_SAVE_DIR, f'edge_{i}.png'), 
                edge_img, 
                cmap='gray'
            )

    def forward(self, backbone_features, input_tensor, targets=None):
        input_size = input_tensor.size()
        
        # Process backbone features
        stage_features = {
            'stage3': self.feature_transform_256(backbone_features[2]),
            'stage4': self.feature_transform_512(backbone_features[3]),
            'stage7': self.feature_transform_4096(backbone_features[6])
        }
        
        # Generate edge predictions
        edge_preds = [
            F.interpolate(self.edge_branches[name](feat), input_size[2:], 
                         mode='bilinear', align_corners=True)
            for name, feat in stage_features.items()
        ]
        
        # Canny edge detection
        numpy_images = input_tensor.cpu().numpy().transpose(0,2,3,1).astype(np.uint8)
        canny_edges = np.zeros((input_size[0], 1, input_size[2], input_size[3]))
        for i in range(input_size[0]):
            canny_edges[i] = cv2.Canny(numpy_images[i], 10, 100)
        canny_tensor = torch.from_numpy(canny_edges).float().to(input_tensor.device)
        
        # Save detected edges
        self._save_edge_detection(canny_tensor, input_size[0])
        
        # Feature integration
        integrated_features = self.multiscale_processors(
            F.interpolate(backbone_features[0], input_size[2:], 
                         mode='bilinear', align_corners=True)
        )
        
        # Final outputs
        edge_output = torch.sigmoid(integrated_features)
        combined_output = torch.sigmoid(
            nn.Conv2d(2, 1, 1)(torch.cat([edge_output, canny_tensor], 1))
        )
        
        return (
            self.loss_calculator(combined_output, targets),
            self.edge_feature_extractor(combined_output)
        )
