import torch
from torch import nn
from torch.nn import functional as F
      
def standardize(xi, method='z', epsilon=1e-6):
    
    if method=='z':
        xi = (xi - torch.mean(xi, dim=(-2, -1), keepdim=True)) / (torch.std(xi, dim=(-2, -1), keepdim=True) + epsilon)
    else:
        print('Not implemented standardization method.')
    
    return xi

def get_sobel_kernels():
    
    """Creates a pair of 2D Sobel kernels."""
    
    kernel_dx = torch.tensor([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
    kernel_dy = torch.tensor([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
    
    return kernel_dx, kernel_dy

def get_uniform_kernel(kernel_size=int):
    
    """Creates a 2D uniform kernel (box blur)."""
    
    kernel = torch.ones(kernel_size, kernel_size)
    
    return kernel / kernel.sum()

def get_gaussian_kernel(kernel_size: int, sigma: float = 1.0):
    
    """Creates a 2D Gaussian kernel."""
    
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    return kernel / kernel.sum()

def pad_finite_difference(phi: torch.Tensor, pad_width: tuple) -> torch.Tensor:
    """
    Pads a 4D tensor using iterative first-order finite difference extrapolation.

    Args:
        phi (torch.Tensor): The input tensor to pad, with shape (B, C, H, W).
        pad_width (tuple): A tuple of 4 integers specifying the padding for the
                           last two dimensions: (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        torch.Tensor: The padded tensor.
    """
    pad_left, pad_right, pad_top, pad_bottom = pad_width
    
    # Create a new tensor with the padded size (initially with zeros)
    padded_phi = F.pad(phi, pad_width)
    
    # --- Step 1: Pad Left and Right (Horizontal Extrapolation) ---
    v_slice = slice(pad_top, -pad_bottom if pad_bottom > 0 else None)

    # Pad Left iteratively
    for i in range(pad_left):
        col_to_fill = pad_left - 1 - i
        col_source1 = col_to_fill + 1
        col_source2 = col_to_fill + 2
        padded_phi[..., v_slice, col_to_fill] = \
            2 * padded_phi[..., v_slice, col_source1] - padded_phi[..., v_slice, col_source2]

    # Pad Right iteratively
    for i in range(pad_right):
        col_to_fill = padded_phi.shape[-1] - pad_right + i
        col_source1 = col_to_fill - 1
        col_source2 = col_to_fill - 2
        padded_phi[..., v_slice, col_to_fill] = \
            2 * padded_phi[..., v_slice, col_source1] - padded_phi[..., v_slice, col_source2]

    # --- Step 2: Pad Top and Bottom (Vertical Extrapolation) ---
    # This step uses the already horizontally-padded tensor.

    # Pad Top iteratively
    for i in range(pad_top):
        row_to_fill = pad_top - 1 - i
        row_source1 = row_to_fill + 1
        row_source2 = row_to_fill + 2
        padded_phi[..., row_to_fill, :] = \
            2 * padded_phi[..., row_source1, :] - padded_phi[..., row_source2, :]
            
    # Pad Bottom iteratively
    for i in range(pad_bottom):
        row_to_fill = padded_phi.shape[-2] - pad_bottom + i
        row_source1 = row_to_fill - 1
        row_source2 = row_to_fill - 2
        padded_phi[..., row_to_fill, :] = \
            2 * padded_phi[..., row_source1, :] - padded_phi[..., row_source2, :]

    return padded_phi