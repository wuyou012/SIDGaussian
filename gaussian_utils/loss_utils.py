#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import random


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def get_vit_feature(x, ext):
    """
    :x: shape(batch, channel, height, width)
    """
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=x.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                        device=x.device).reshape(1, 3, 1, 1)
    x = F.interpolate(x, size=(224, 224))
    x = (x - mean) / std
    return ext.get_feature_from_input(x)[-1][0, 0, :]

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_partition2(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    
    # Calculate the new height and width that are divisible by window_size
    new_H = H - (H % window_size)
    new_W = W - (W % window_size)
    
    # Trim the tensor to the new dimensions
    x = x[:, :, :new_H, :new_W]
    
    # Reshape into windows
    x = x.view(B, C, new_H // window_size, window_size, new_W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    
    return windows

def window_partition3(x, window_size, step_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
        step_size (int): step size for sliding window

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    # Calculate the number of windows along height and width
    num_windows_h = (H - window_size) // step_size + 1
    num_windows_w = (W - window_size) // step_size + 1
    
    # Allocate memory for windows
    windows = x.new_zeros(B, C, num_windows_h, num_windows_w, window_size, window_size)
    
    # Slide the window over the input tensor
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            y0 = i * step_size
            x0 = j * step_size
            windows[:, :, i, j] = x[:, :, y0:y0+window_size, x0:x0+window_size]
    
    # Reshape the windows tensor
    windows = windows.view(-1, C, window_size, window_size)
    
    return windows

def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
            (weight_x.sum() + weight_y.sum())
    return loss

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def normalize0(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=0, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=0, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def random_crop(tensor, crop_size):

    c, h, w = tensor.shape

    crop_height, crop_width = crop_size
    if crop_height > h or crop_width > w:
        raise ValueError("Crop size must be less than or equal to the original size.")

    top = random.randint(0, h - crop_height)
    left = random.randint(0, w - crop_width)

    cropped_tensor = tensor[:, top:top + crop_height, left:left + crop_width]
    return cropped_tensor

def random_crop1(gt_tensor, re_tensor, crop_size):

    _, h, w = gt_tensor.shape
    _, h_re, w_re = re_tensor.shape

    crop_height, crop_width = crop_size
    if crop_height > h or crop_width > w:
        raise ValueError("Crop size must be less than or equal to the original size.")

    top = random.randint(0, h - crop_height)
    left = random.randint(0, w - crop_width)

    cropped_gt_tensor = gt_tensor[:, top:top + crop_height, left:left + crop_width]

    re_crop_height = 2 * crop_height
    re_crop_width = 2 * crop_width

    re_top = max(top - crop_height // 2, 0)  # 确保不超出上边界
    re_left = max(left - crop_width // 2, 0)  # 确保不超出左边界

    re_bottom = min(re_top + re_crop_height, h_re)
    re_right = min(re_left + re_crop_width, w_re)

    if re_bottom - re_top < re_crop_height:
        re_top = re_bottom - re_crop_height
    if re_right - re_left < re_crop_width:
        re_left = re_right - re_crop_width

    cropped_re_tensor = re_tensor[:, re_top:re_bottom, re_left:re_right]
    # cropped_re_tensor = re_tensor[:, top - 0.5*crop_height:top + 1.5*crop_height, left - 0.5*crop_width:left + 1.5*crop_width]
    # print('gt',top,top + crop_height,left,left + crop_width)
    # print('re',re_top, re_bottom, re_left, re_right)
    return cropped_gt_tensor, cropped_re_tensor
def random_crop3(gt_tensor, re_tensor, crop_size):
    _, h, w = gt_tensor.shape
    _, h_re, w_re = re_tensor.shape

    # Assert that both tensors have the same dimensions
    assert (h == h_re) and (w == w_re), f"Image size mismatch: gt_tensor is {h}x{w}, re_tensor is {h_re}x{w_re}"

    # Ensure crop size does not exceed the image size
    crop_height = min(crop_size[0], h)
    crop_width = min(crop_size[1], w)

    # Generate random top-left corner for the crop within the valid range
    top = torch.randint(0, h - crop_height + 1, (1,)).item()
    left = torch.randint(0, w - crop_width + 1, (1,)).item()

    # Crop both tensors
    gt_cropped = gt_tensor[:, top:top + crop_height, left:left + crop_width]
    re_cropped = re_tensor[:, top:top + crop_height, left:left + crop_width]

    return gt_cropped, re_cropped
def random_crop0(tensor, crop_size):

    b, c, h, w = tensor.shape

    crop_height, crop_width = crop_size
    if crop_height > h or crop_width > w:
        raise ValueError("Crop size must be less than or equal to the original size.")

    top = random.randint(0, h - crop_height)
    left = random.randint(0, w - crop_width)

    cropped_tensor = tensor[:, :, top:top + crop_height, left:left + crop_width]
    return cropped_tensor

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

class PatchSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, stride=1, num_proj=256, c=3, l2=False):
        super(PatchSWDLoss, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_proj = num_proj
        self.l2 = l2
        self.c = c
        self.sample_projections()

    def sample_projections(self):
        # Sample random normalized projections
        rand = torch.randn(self.num_proj, self.c*self.patch_size**2) # (slice_size**2*ch)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # noramlize to unit directions
        self.rand = rand.reshape(self.num_proj, self.c, self.patch_size, self.patch_size)

    def forward(self, x, y, reset_projections=True):
        if reset_projections:
            self.sample_projections()
        self.rand = self.rand.to(x.device)
        # Project patches
        projx = F.conv2d(x, self.rand).transpose(1,0).reshape(self.num_proj, -1)
        projy = F.conv2d(y, self.rand).transpose(1,0).reshape(self.num_proj, -1)

        # Duplicate patches if number does not equal
        projx, projy = duplicate_to_match_lengths(projx, projy)

        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        if self.l2:
            loss = ((projx - projy)**2).mean()
        else:
            loss = torch.abs(projx - projy).mean()

        return loss


def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2