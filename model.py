import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class DWSConv(nn.Module):
    """
    Depthwise Separable Convolution with BatchNorm and Activation.
    
    Attributes:
        depthwise (nn.Conv2d): Depthwise convolution layer.
        pointwise (nn.Conv2d): Pointwise convolution layer.
        bn (nn.BatchNorm2d): Batch normalization.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation (SiLU).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True):
        """
        Initialize the DWSConv layer.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding. Defaults to autopad.
            d (int): Dilation.
            act (bool | nn.Module): Activation type.
        """
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Forward pass applying depthwise → pointwise → BN → activation.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class CondConv(nn.Module):
    """
    Conditional Convolution Layer with multiple expert kernels and input-dependent routing.

    Attributes:
        expert_weights (nn.Parameter): Learnable expert kernels.
        expert_bias (nn.Parameter): Learnable expert biases.
        routing_fn (nn.Module): Learns expert weights based on input.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True, num_experts=4):
        """
        Initialize CondConv layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding. Uses autopad if None.
            d (int): Dilation.
            act (bool | nn.Module): Activation.
            num_experts (int): Number of expert convolution kernels.
        """
        super().__init__()
        self.k = k
        self.s = s
        self.d = d
        self.p = autopad(k, p, d)
        self.num_experts = num_experts

        # Expert weights and bias
        self.expert_weights = nn.Parameter(torch.randn(num_experts, c2, c1, k, k))
        self.expert_bias = nn.Parameter(torch.randn(num_experts, c2))

        # Routing function (global avg pooling + FC)
        self.routing_fn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, num_experts),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        B = x.size(0)
        routing_weights = self.routing_fn(x)  # (B, num_experts)

        # Weighted combination of expert weights and biases
        weight = torch.einsum('be,eocij->bocij', routing_weights, self.expert_weights)  # (B, c2, c1, k, k)
        bias = torch.einsum('be,eo->bo', routing_weights, self.expert_bias)             # (B, c2)

        # Apply sample-wise convolution
        outputs = []
        for i in range(B):
            xi = x[i:i+1]
            wi = weight[i]
            bi = bias[i]
            yi = F.conv2d(xi, wi, bi, stride=self.s, padding=self.p, dilation=self.d)
            outputs.append(yi)

        out = torch.cat(outputs, dim=0)  # (B, c2, H_out, W_out)
        return self.act(self.bn(out))

class SE(nn.Module):
    """
    Squeeze-and-Excitation (SE) block using Conv2d for channel recalibration.
    """

    def __init__(self, c1, reduction=16):
        super().__init__()
        c_ = max(1, c1 // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DeformableConv(nn.Module):
    """
    Deformable Convolution block with offset learning and activation.

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
        k (int): Kernel size.
        s (int): Stride.
    """

    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * k * k, kernel_size=k, stride=s, padding=k // 2)
        self.deform_conv = DeformConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        return self.act(self.bn(x))
    
class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    def __init__(self, c1, c2=None):  # ✅ Add c2 for compatibility
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x