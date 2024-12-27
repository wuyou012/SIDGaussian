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

import os
import random
import json
from gaussian_utils.system_utils import searchForMaxIteration
from gaussian_utils.graphics_utils import fov2focal
from gaussian_scene.dataset_readers import sceneLoadTypeCallbacks
from gaussian_scene.gaussian_model import GaussianModel
from gaussian_arguments import ModelParams
from gaussian_utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
from scipy.spatial import KDTree

from scipy.spatial.transform import Rotation as R

def readColmapCamera(path, path_help, images, eval):
    scene_info = sceneLoadTypeCallbacks["Colmap"](path, path_help, images, eval)
    # camlist = []
    # if scene_info.test_cameras:
    #     camlist.extend(scene_info.test_cameras)
    # if scene_info.train_cameras:
    #     camlist.extend(scene_info.train_cameras)
    return scene_info.all_cameras

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, extended_pcd=None, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.params_vars = {'rot': 0.09, 'trans': 0.09}

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        path_help = ""
        if os.path.exists(os.path.join(args.source_path, path_help, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, path_help, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif extended_pcd is not None:
            self.gaussians.create_from_pcd(extended_pcd, self.cameras_extent)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        # self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_ply(f"{point_cloud_path}_point_cloud.ply")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def increaseVariance(self):
        self.params_vars['rot'] += 0.01
        self.params_vars['trans'] += 0.01


    def drawRandomRT(self):
        randomR = np.random.normal(loc=0, scale=np.sqrt(self.params_vars['rot']))
        randomT = np.random.normal(loc=0, scale=np.sqrt(self.params_vars['trans']))
        # print(f"R: {randomR}, T: {randomT}\n")
        return {"R":randomR, "T":randomT}
    
    def getNormPrincipalRay(self, cam_info):
        """
        In 3D camera vision, the ray that is perpendicular to the focal 
        plane is often referred to as the principal ray or the optical 
        axis of the camera1. This ray passes through the camera center 
        (also known as the optical center or pinhole) and the principal 
        point, which is the point where the principal ray intersects the 
        image plane1.
        """
        # Given R, T, FoVx, and FoVy
        R = cam_info.R  # 3x3 rotation matrix
        T = cam_info.T  # 3x1 translation vector
        FoVx = cam_info.FoVx  # Field of view angle in x direction
        FoVy = cam_info.FoVy  # Field of view angle in y direction
        width = cam_info.image_width
        height = cam_info.image_height

        # Calculate focal length using fov2focal function
        f_x = fov2focal(width, FoVx)
        f_y = fov2focal(height, FoVy)

        # Principal point is typically at the center of the image
        c_x = width / 2
        c_y = height / 2

        # Construct the intrinsic matrix K
        K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])

        # Calculate camera center in world coordinates
        C = -np.dot(R.T, T)

        # Principal point in image coordinates (center of the image)
        p_image = np.array([c_x, c_y, 1])

        # Camera projection matrix
        P = np.dot(K, np.hstack((R, T.reshape(-1, 1))))

        # Calculate world coordinates of principal point using pseudo-inverse
        P_pseudo_inv = np.linalg.pinv(P)
        p_world_homogeneous = np.dot(P_pseudo_inv, p_image)

        # Normalize to ensure the last component is 1
        p_world = p_world_homogeneous[:-1] / p_world_homogeneous[-1]

        # Calculate and normalize principal ray direction
        NormPrincipalRay = C - p_world
        NormPrincipalRay /= np.linalg.norm(NormPrincipalRay)

        return NormPrincipalRay

    def initSideViewSystem1(self, scale=1.0):
        """
        :TODO: Loop over the entire training set and extract the cameras, we assume that the viewangles are sorted.
        """
        train_cameras = self.getTrainCameras(scale=scale)

        uids = []
        for cam in train_cameras:
            uids.append([cam.uid])

        angles = np.full((len(uids), len(uids)), 180, dtype=float)
        for i, uid1 in enumerate(uids):
            for j, uid2 in enumerate(uids):
                if uid1 == uid2:
                    continue
                r0 = train_cameras[i].R
                r1 = train_cameras[j].R
                rot_mat_rel = np.matmul(np.transpose(r0), r1)

                # Create Rotation objects from the rotation matrices
                r0 = R.from_matrix(r0)
                r1 = R.from_matrix(r1)
                r_rel = R.from_matrix(rot_mat_rel)

                # Calculate the relative angle between r0 and r1
                angle = r0.inv() * r_rel * r1
                angles[i, j] = angle.magnitude()

        min_indices = np.argmin(angles, axis=1)

        return_dict = {}

        for i, uid1 in enumerate(uids):
            hash_key = uid1
            return_dict[hash_key[0]] = {'next': train_cameras[min_indices[i]], 'prev': train_cameras[min_indices[i]], 'this': train_cameras[i]}

        return return_dict

    def initSideViewSystem(self, scale=1.0):
        """
        :TODO: Loop over the entire training set and extract the cameras, we assume that the viewangles are sorted.
        """
        train_cameras = self.getTrainCameras(scale=scale)

        uids = []
        for i, cam in enumerate(train_cameras):
            uids.append([cam.uid])
    
        # Create a dictionary where each key is a UID and each value is another dictionary
        # containing the UIDs of the previous and next rays in the loop
        loop_dict = {}
        for i in range(1, len(uids) - 1):
            hash_key = uids[i]
            loop_dict[hash_key[0]] = {'prev': train_cameras[i - 1], 'next': train_cameras[i + 1], 'this': train_cameras[i]}
        
        # Handle the first and last rays in the loop
        loop_dict[uids[0][0]] = {'prev': train_cameras[-1], 'next': train_cameras[1], 'this': train_cameras[0]}
        loop_dict[uids[-1][0]] = {'prev': train_cameras[-2], 'next': train_cameras[0], 'this': train_cameras[-1]}

        return loop_dict
    
    def getRotationMatrix(self, d, cam_info):
        """
        Calculate the rotation matrix from the principal ray direction.
        """
        # Given d, T, FoVx, and FoVy
        T = cam_info.T  # 3x1 translation vector
        FoVx = cam_info.FoVx  # Field of view angle in x direction
        FoVy = cam_info.FoVy  # Field of view angle in y direction
        width = cam_info.width
        height = cam_info.height

        # Calculate focal length using fov2focal function
        f_x = fov2focal(width, FoVx)
        f_y = fov2focal(height, FoVy)

        # Principal point is typically at the center of the image
        c_x = width / 2
        c_y = height / 2

        # Construct the intrinsic matrix K
        K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])

        # Calculate camera center in world coordinates
        C = -np.dot(K.T, T)

        # Principal point in image coordinates (center of the image)
        p = np.array([FoVx / 2, FoVy / 2, 1])

        # Camera projection matrix
        P = np.dot(K, np.hstack((np.eye(3), T)))

        # Calculate world coordinates of principal point
        P = np.dot(np.linalg.inv(P), p)

        # Calculate and normalize principal ray direction
        d_prime = P - C
        d_prime /= np.linalg.norm(d_prime)

        # Calculate rotation matrix
        R = np.dot(d, d_prime.T)

        return R
