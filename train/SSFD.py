import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SSFDLoss(nn.Module):
    """
    Knowledge distillation loss with inter-layer and intra-layer feature matching
    Features:
    - Inter-layer feature discrepancy: Aligns intermediate features between student and teacher
    - Intra-layer feature consistency: Enforces similarity between different channels within same layer
    """

    def __init__(self, lambda_weight):
        super().__init__()
        self.lambda_weight = lambda_weight

    def _align_features(self, student_feat, teacher_feat):
        """Align spatial dimensions between features using adaptive pooling"""
        s_h, t_h = student_feat.shape[2], teacher_feat.shape[2]
        if s_h != t_h:
            target_size = (t_h, t_h) if s_h > t_h else (s_h, s_h)
            return F.adaptive_avg_pool2d(student_feat, target_size)
        return student_feat

    def inter_layer_loss(self, student_feat, teacher_feat):
        """Calculate inter-layer feature discrepancy with channel sampling"""
        # Align spatial dimensions
        aligned_student = self._align_features(student_feat, teacher_feat)
        
        # Random channel sampling
        min_channels = min(student_feat.shape[1], teacher_feat.shape[1])
        student_channels = random.sample(range(student_feat.shape[1]), min_channels)
        teacher_channels = random.sample(range(teacher_feat.shape[1]), min_channels)
        
        # Calculate MSE between selected channels
        return F.mse_loss(
            aligned_student[:, student_channels, :, :],
            teacher_feat[:, teacher_channels, :, :].detach()
        )

    def intra_layer_loss(self, features):
        """Calculate intra-layer feature consistency"""
        # Normalize and sort features by channel importance
        normalized_feat = F.normalize(features, p=2, dim=(2,3))
        channel_importance = normalized_feat.mean(dim=[0,2,3])
        sorted_indices = torch.argsort(channel_importance, descending=True)
        
        sorted_feat = torch.index_select(features, 1, sorted_indices)
        split_point = sorted_feat.shape[1] // 2
        
        # Split channels and calculate consistency
        upper_half = sorted_feat[:, :split_point, :, :]
        lower_half = sorted_feat[:, split_point:, :, :]
        return F.mse_loss(upper_half, lower_half)

    def _process_features(self, raw_features):
        """Process feature tensor dimensions"""
        # Convert from [batch, height, width, channels] to [batch, channels, height, width]
        return [feat.permute(0, 3, 1, 2).unsqueeze(0) for feat in raw_features]

    def forward(self, student_features, teacher_features):
        # Preprocess feature dimensions
        processed_features = self._process_features(student_features)
        
        # Extract hierarchical features
        final_features = [layer[-1] for layer in processed_features]  # Deep features
        initial_features = [layer[0] for layer in processed_features]  # Shallow features

        # Calculate intra-layer losses
        intra_loss = sum(self.intra_layer_loss(feat) for feat in final_features + initial_features)
        intra_loss /= len(final_features + initial_features)

        # Calculate inter-layer losses
        inter_loss = sum(self.inter_layer_loss(feat, teacher_features) for feat in final_features + initial_features)
        inter_loss /= len(final_features + initial_features)

        # Combine losses
        total_loss = (intra_loss + inter_loss) * self.lambda_weight
        return total_loss
