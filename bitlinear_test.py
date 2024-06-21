import torch
import torch.nn.functional as F
from torch import Tensor, nn


def activation_quant(x):
    """Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
    x: an activation tensor with shape [n, d]
    Returns:
    y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
    w: a weight tensor with shape [d, k]
    Returns:
    u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/pdf/1910.07467.pdf.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor to normalize

        Returns:
            Tensor: The output tensor after applying RMSNorm.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return x_normed * self.scale


class BitLinear(nn.Linear):
    """
    This is only for training, and kernel optimization is needed for efficiency.
    """

    def forward(self, x):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        # w = self.weight # a weight tensor with shape [d, k]
        w = torch.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="float32"))
        x_norm = RMSNorm(self.in_features)(x)
        # y = F.linear(x_norm, w)

        # A trick for implementing Straight-Through-Estimator (STE) using detach()
        # x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        x_quant = x_norm + activation_quant(x_norm).detach() - x_norm.detach()
        w_quant = w + weight_quant(w).detach() - w.detach()
        # x_quant = x_norm
        # w_quant = w
        y = F.linear(x_quant, w_quant)

        # print(torch.gradient(x_norm))
        # print(torch.gradient(w))
        # print()

        # print(torch.gradient(x_quant))
        # print(torch.gradient(w_quant))
        # print(torch.gradient(y))

        return y


if __name__ == "__main__":
    import numpy as np

    # Input
    x = torch.tensor(np.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]], dtype="float32"), requires_grad=True)

    # BitLinear layer
    layer = BitLinear(3, 4)

    # Output
    y = layer(x)
    print()
    print(y)
    print()
    y.backward(torch.ones(2, 4), retain_graph=True)
    print(x)
    print(x.grad)
