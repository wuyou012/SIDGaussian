#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

###############################################################################
#                              Imports & Globals                              #
###############################################################################
import os
import sys
import uuid
import json
import random
import imageio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # Will be disabled if not found
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics import PearsonCorrCoef
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt

# Local imports from your repository or modules
import gaussian_utils.loss_utils as loss_utils
from gaussian_utils.loss_utils import (
    l1_loss,
    l1_loss_mask,
    l2_loss,
    ssim,
    get_vit_feature
)
from gaussian_utils.depth_utils import estimate_depth
from gaussian_utils.general_utils import safe_state
from gaussian_utils.image_utils import psnr
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from gaussian_utils.extractor import VitExtractor

# Optional: Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


###############################################################################
#                      Utility Functions & Helper Methods                     #
###############################################################################
def prepare_output_and_logger(args):
    """
    Create the output folder and a tensorboard logger if available.
    """
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    
    # Log the parser arguments
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_func,
    testing_iterations,
    scene: Scene,
    render_func,
    render_args,
    metrics
):
    """
    Logs training metrics to tensorboard (if available) and runs tests 
    on specified iterations.
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Run test/evaluation at certain iterations
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': scene.getTrainCameras()}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render_func(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # Log a few images
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/render",
                            image[None],
                            global_step=iteration
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None],
                                global_step=iteration
                            )
                    
                    # Compute metrics
                    l1_test += l1_func(image, gt_image).mean().double()
                    _psnr = psnr(image, gt_image, None).mean().double()
                    _ssim = ssim(image, gt_image, None).mean().double()
                    _lpips = lpips(image, gt_image, None, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips

                # Average across all cameras
                n_cams = len(config['cameras'])
                l1_test /= n_cams
                psnr_test /= n_cams
                ssim_test /= n_cams
                lpips_test /= n_cams

                # Print and log the metrics
                print(
                    f"\n[ITER {iteration}] Evaluating {config['name']}: "
                    f"L1 {l1_test:.4f} PSNR {psnr_test:.4f} SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f}"
                )
                if config['name'] == 'test':
                    metrics['PSNR'].append(float(f"{psnr_test.item():.4f}"))
                    metrics['SSIM'].append(float(f"{ssim_test.item():.4f}"))
                    metrics['LPIPS'].append(float(f"{lpips_test.item():.4f}"))

                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        # Additional logging
        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


###############################################################################
#                             Main Training Loop                               #
###############################################################################
def training(dataset, opt, pipe, args, metrics):
    """
    Main training function that handles:
      - Model creation & checkpoint loading
      - Iteration loop, including forward pass, loss calculation
      - Logging, saving, & evaluations
    """
    # Unpack arguments
    testing_iterations    = args.test_iterations
    saving_iterations     = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint            = args.start_checkpoint
    debug_from            = args.debug_from
    first_iter            = 0

    # Prepare logger and model
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)

    # Load checkpoint if provided
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Scene background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Progress bar
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    ema_loss_for_log = 0.0
    first_iter += 1

    # Initialize viewpoint and pseudo stacks
    viewpoint_stack, pseudo_stack = None, None

    # Set up ViT extractor
    vit_ext0 = VitExtractor(model_name='dino_vits16', device="cuda:0")

    # ------------------------------- Training Loop -------------------------------
    for iteration in range(first_iter, opt.iterations + 1):
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Increase spherical harmonics degree every 500 iterations
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera from the training set
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Forward pass: render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Compute losses
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss_mask(image, gt_image)
        loss_l1_ssim = ((1.0 - opt.lambda_dssim) * Ll1 +
                        opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        loss = loss_l1_ssim

        # Depth losses
        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
            (1 - pearson_corrcoef(-midas_depth, rendered_depth)),
            (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        loss_depth = args.depth_weight * depth_loss
        loss += loss_depth

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001

        # Handle pseudo-cameras for additional depth supervision
        flag_pseudo = False
        if (iteration % args.sample_pseudo_interval == 0
            and args.start_sample_pseudo < iteration < args.end_sample_pseudo):
            
            # Pick a random camera from the pseudo set
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            
            # Render pseudo camera & estimate pseudo depth
            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            # Convert to shape [B=1, C=1, H, W] for window partition
            rdp4 = rendered_depth_pseudo.unsqueeze(0).unsqueeze(0)
            mdp4 = midas_depth_pseudo.unsqueeze(0).unsqueeze(0)
            rendered_pseudo_windows = loss_utils.window_partition3(rdp4, 126, 64)
            midas_pseudo_windows = loss_utils.window_partition3(mdp4, 126, 64)

            # Flatten for global depth correlation
            rendered_depth_pseudo0 = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo0 = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo0, -midas_depth_pseudo0)).mean()

            # Randomly select a window patch to check local correlation
            rendered_pseudo_windows = rendered_pseudo_windows.squeeze(1)
            midas_pseudo_windows = midas_pseudo_windows.squeeze(1)
            rn = random.randint(0, rendered_pseudo_windows.shape[0] - 1)
            rendered_pseudo_windows = rendered_pseudo_windows[rn].reshape(-1, 1)
            midas_pseudo_windows = midas_pseudo_windows[rn].reshape(-1, 1)

            # Local normalization
            nrpw0 = loss_utils.normalize0(rendered_pseudo_windows)
            nmpw0 = loss_utils.normalize0(midas_pseudo_windows)
            patch_depth_loss_pseudo = (1 - pearson_corrcoef(nrpw0, -nmpw0)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500.0, 1.0)
                loss_depth_pseudo = loss_scale * args.N * (depth_loss_pseudo + args.W * patch_depth_loss_pseudo)
                loss += loss_depth_pseudo

            # DINO feature matching
            rendered_image_pseudo = render_pkg_pseudo["render"]
            gt_crop, rendered_crop = loss_utils.random_crop1(gt_image, rendered_image_pseudo, (84, 63))
            render_crop_pseudo_vit = get_vit_feature(rendered_crop.unsqueeze(0), vit_ext0)
            gt_crop_vit = get_vit_feature(gt_crop.unsqueeze(0), vit_ext0)
            side_loss_dino = F.mse_loss(render_crop_pseudo_vit, gt_crop_vit)
            loss_dino = args.D * side_loss_dino
            loss += loss_dino

            flag_pseudo = True

        # Backprop
        loss.backward()
        torch.cuda.empty_cache()

        with torch.no_grad():
            # Update progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "points": f"{scene.gaussians.get_xyz.shape[0]}",
                    "L1": f"{loss_l1_ssim:.4f}",
                    "depth": f"{loss_depth:.4f}",
                    "d_pse": f"{depth_loss_pseudo:.4f}" if flag_pseudo else "X.XXXX",
                    "dino": f"{loss_dino:.4f}" if flag_pseudo else "X.XXXX",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging / Reporting
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                testing_iterations,
                scene,
                render,
                (pipe, background),
                metrics
            )

            # Save results
            if iteration > first_iter and (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, f"chkpnt{iteration}.pth")
                )

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (iteration > opt.densify_from_iter and
                        iteration % opt.densification_interval == 0):
                    size_threshold = None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.prune_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        iteration
                    )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Update learning rate & reset opacity if needed
            gaussians.update_learning_rate(iteration)
            if ((iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 
                and iteration > args.start_sample_pseudo):
                gaussians.reset_opacity()


###############################################################################
#                              Main Entry Point                                #
###############################################################################
if __name__ == "__main__":
    # Parse CLI arguments
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7000, 8000, 9000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--W", type=float, default=0.5) # window
    parser.add_argument("--D", type=float, default=0.8) # dino
    parser.add_argument("--N", type=float, default=1.0) # total depth

    args = parser.parse_args(sys.argv[1:])
    print('args:', args.W, args.D, args.N)

    scene_data_dir = lp.extract(args).source_path
    scene_name = scene_data_dir.split('/')[-1]

    print(args.test_iterations)
    print("Optimizing " + args.model_path)

    metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Enable anomaly detection if requested
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Run training
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args,
        metrics
    )

    print("\nTraining complete.")

    # Saving scores
    output_dir = os.path.dirname(args.model_path)
    output_data = {
        "PSNR": metrics['PSNR'],
        "SSIM": metrics['SSIM'],
        "LPIPS": metrics['LPIPS'],
        "scene": scene_name
    }
    output_path = os.path.join(output_dir, "test_results.json")
    with open(output_path, "a+") as f:
        json.dump(output_data, f, indent=4)
