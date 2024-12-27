import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
import tqdm
from torchvision import transforms as T

from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Slerp

# from .ray_utils import *
from einops import rearrange

import cv2


def assign_last(a, index, b):
    """a[index] = b
    """
    index = index[::-1]
    b = b[::-1]

    ix_unique, ix_first = np.unique(index, return_index=True)
    # np.unique: return index of first occurrence.
    # ix_unique = index[ix_first]

    a[ix_unique] = b[ix_first]
    return a


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    # x = depth.cpu().numpy()
    x = depth
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    return x_


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

def trans_v(vector):
    """
    Create a translation matrix for a given vector.

    Parameters:
    vector (np.array): A NumPy array representing the translation vector.
    vector = np.array([x, y, z])

    Returns:
    np.array: A 4x4 NumPy array representing the translation matrix.
    """
    # Ensure the vector is a NumPy array
    vector = np.array(vector)
    
    # Check if the vector has three components
    if vector.shape[0] != 3:
        raise ValueError("The translation vector must have three components.")
    
    # Create the translation matrix
    translation_matrix = np.array([
        [1, 0, 0, vector[0]],
        [0, 1, 0, vector[1]],
        [0, 0, 1, vector[2]],
        [0, 0, 0, 1]
    ])
    
    return translation_matrix


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rot_z(th): return torch.Tensor([
    [np.cos(th), -np.sin(th), 0, 0],
    [np.sin(th), np.cos(th), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(c2w, theta, phi):
    # c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi)  # @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def rotate(angle):
    return rot_z(angle/180.*np.pi)
    # return rot_theta(angle/180.*np.pi)
    # return rot_phi(angle/180.*np.pi)


def flatten(pose):
    if pose.shape[0] != 4:
        pose = torch.cat([pose, torch.Tensor([[0, 0, 0, 1]])], dim=0)
    return torch.inverse(pose)[:3, :4]

# llff uses right up back
# opencv: right, down, forward

def rotate_3d(c2w, x, y, z):
    """
    TODO: to perform a 3d rotation
    """
    rot = rot_phi(x/180.*np.pi) @ rot_theta(y/180.*np.pi) @ rot_z(z/180.*np.pi)
    return rot @ c2w

def relative_rot(pose1, pose2):
    """
    TODO: Calculate the relative rotation matrix between two poses.

    Parameters:
    pose1 (np.array): A 4x4 NumPy array representing the first pose matrix.
    pose2 (np.array): A 4x4 NumPy array representing the second pose matrix.

    Returns:
    np.array: A 3x3 NumPy array representing the relative rotation matrix.
    """
    # Extract the rotation matrices (top-left 3x3 submatrix)
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    
    # Calculate the inverse of the first rotation matrix
    R1_inv = np.linalg.inv(R1)
    
    # Calculate the relative rotation matrix
    relative_rotation_matrix = R1_inv @ R2
    
    return relative_rotation_matrix

def rot2quat(rotation_matrix):
    """
    TODO: Convert a rotation matrix to a quaternion.

    Parameters:
    rotation_matrix (np.array): A 3x3 or 4x4 NumPy array representing the rotation matrix.

    Returns:
    np.array: A NumPy array representing the quaternion.
    """
    # Ensure the input is a NumPy array
    rotation_matrix = np.array(rotation_matrix)
    
    # Check if the input is a 3x3 or 4x4 matrix
    if rotation_matrix.shape == (3, 3):
        # Convert the 3x3 rotation matrix to a quaternion
        r = R.from_matrix(rotation_matrix)
    elif rotation_matrix.shape == (4, 4):
        # Extract the top-left 3x3 submatrix if the input is a 4x4 matrix
        r = R.from_matrix(rotation_matrix[:3, :3])
    else:
        raise ValueError("The rotation matrix must be 3x3 or 4x4.")
    
    # Return the quaternion
    return r.as_quat()

def quat2rot(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    quaternion (np.array): A NumPy array representing the quaternion.

    Returns:
    np.array: A 3x3 NumPy array representing the rotation matrix.
    """
    # Ensure the input is a NumPy array
    quaternion = np.array(quaternion)
    
    # Check if the input is a valid quaternion
    if quaternion.shape[0] != 4:
        raise ValueError("The input must be a quaternion with four components.")
    
    # Convert the quaternion to a rotation object
    rotation = R.from_quat(quaternion)
    
    # Return the rotation matrix
    return rotation.as_matrix()

def slerp(q1, q2, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.

    Parameters:
    q1 (np.array): The first quaternion as a NumPy array.
    q2 (np.array): The second quaternion as a NumPy array.
    t (float): The interpolation parameter between 0 and 1.

    Returns:
    np.array: The interpolated quaternion.
    """
    # Convert the quaternions to scipy Rotation objects
    # r1 = R.from_quat(q1)
    # r2 = R.from_quat(q2)
    t = abs(t)
    if t > 1:
        t = 1

    key_rots = R.from_quat([q1, q2])
    key_times = [0, 1]

    slerp = Slerp(key_times, key_rots)
    times_to_interpolate = [t]
    
    interpolated_rotation = slerp(times_to_interpolate)[0]

    # Perform SLERP
    # interpolated_rotation = Slerp([0, 1], key_rots)
    
    # Return the interpolated quaternion
    return interpolated_rotation.as_quat()


def convert(c2w, scale_factor=1):
    # return np.linalg.inv(c2w)
    R, T = c2w[:3, :3], c2w[:3, 3:]
    # T *= scale_factor
    ww = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])
    #  [0, 0, 0, 1]])
    R_ = R.T
    T_ = -1 * R_ @ T
    R_ = ww @ R_
    T_ = ww @ T_
    # print(R_.shape, T_.shape)
    new = np.concatenate((R_, T_), axis=1)
    # new = torch.inverse(torch.from_numpy(ww @ c2w).float())
    new = np.concatenate((new, np.array([[0, 0, 0, 1]])), axis=0)
    return new


def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                                   torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    pts = torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(
        0) * (depth_ref.view(batchsize, -1).unsqueeze(1))

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), pts)

    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                           torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:, :3, :]

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480
    depth_src = K_xyz_src[:, 2:3, :]
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:, 2:3, :] + 1e-9)
    x_src = xy_src[:, 0, :].view([batchsize, height, width])
    y_src = xy_src[:, 1, :].view([batchsize, height, width])

    return x_src, y_src, depth_src
# (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')


def forward_warp(data, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):
    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]
    data = data[0].permute(1, 2, 0)
    new = np.zeros_like(data)
    depth_src = depth_src.reshape(height, width)
    new_depth = np.zeros_like(depth_src)
    yy_base, xx_base = torch.meshgrid([torch.arange(
        0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long)])
    y_res = np.clip(y_res.numpy(), 0, height - 1).astype(np.int64)
    x_res = np.clip(x_res.numpy(), 0, width - 1).astype(np.int64)
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    y_res = y_res.reshape(-1)
    x_res = x_res.reshape(-1)
    # painter's algo
    for i in range(yy_base.shape[0]):
        if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
            new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
            new[y_res[i], x_res[i]] = data[yy_base[i], xx_base[i]]
    return new, new_depth


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    # convert to homogeneous coordinate for faster computation
    pose_avg_homo[:3] = pose_avg
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]),
                       (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row],
                       1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(
        pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        def trans_t(t): return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9*t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        def rot_phi(phi): return np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        def rot_theta(th): return np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],
                       [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        # 36 degree view downwards
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)]
    return np.stack(spheric_poses, 0)