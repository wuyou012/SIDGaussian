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
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
import gaussian_utils.loss_utils as loss_utils
from gaussian_utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, get_vit_feature
from gaussian_utils.depth_utils import estimate_depth
# from gaussian_renderer import render, network_gui
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from gaussian_utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from gaussian_utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from gaussian_utils.extractor import VitExtractor
import torch.nn.functional as F
import random

import imageio

# Uncomment the following line to attach a debugger
# def attach_debugger():
#     import debugpy
#     debugpy.listen(2003)
#     print("Waiting for Debuger to Attach on Port 2003...")
#     debugpy.wait_for_client()
#     print("Attached!")
# attach_debugger()

def training(dataset, opt, pipe, args, metrics):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)

    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1

    vit_ext0 = VitExtractor(model_name='dino_vits16', device="cuda:0")

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
        #                                                                                                        0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss_mask(image, gt_image)
        loss_l1_ssim = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        loss = loss_l1_ssim

        rendered_depth = render_pkg["depth"][0]  # (378 504)
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()

        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
            (1 - pearson_corrcoef(- midas_depth, rendered_depth)),
            (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        loss_depth = args.depth_weight * depth_loss
        loss += loss_depth

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001


        flag_pseudo = False
        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]  # (378,504)
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rdp4 = rendered_depth_pseudo.unsqueeze(0).unsqueeze(0)
            mdp4 = midas_depth_pseudo.unsqueeze(0).unsqueeze(0)
            rendered_pseudo_windows = loss_utils.window_partition3(rdp4, 126,64)
            midas_pseudo_windows = loss_utils.window_partition3(mdp4, 126,64)

            rendered_depth_pseudo0 = rendered_depth_pseudo.reshape(-1, 1)  # [190512, 1]
            midas_depth_pseudo0 = midas_depth_pseudo.reshape(-1, 1)

            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo0, -midas_depth_pseudo0)).mean()  # 0.14

            # ---------------------- add window on pseudo depth ----------------------
            rendered_pseudo_windows = rendered_pseudo_windows.squeeze(1)  # (W*B, H, W)
            midas_pseudo_windows = midas_pseudo_windows.squeeze(1)
            # random select
            rn = random.randint(0, rendered_pseudo_windows.shape[0] - 1)
            rendered_pseudo_windows = rendered_pseudo_windows[rn].reshape(-1, 1)
            midas_pseudo_windows = midas_pseudo_windows[rn].reshape(-1, 1)
            # patch_depth_loss_pseudo = (1 - pearson_corrcoef(rendered_pseudo_windows, -midas_pseudo_windows)).mean()

            # ---------------------- local depth normalization ----------------------
            nrpw0 = loss_utils.normalize0(rendered_pseudo_windows)
            nmpw0 = loss_utils.normalize0(midas_pseudo_windows)
            patch_depth_loss_pseudo = (1 - pearson_corrcoef(nrpw0, -nmpw0)).mean()

            # ---------------------- global depth normalization ----------------------
            # nrpwg = loss_utils.normalize0(rendered_pseudo_windows,std = rendered_depth_pseudo0.std().detach())
            # nmpwg = loss_utils.normalize0(midas_pseudo_windows,std = midas_depth_pseudo0.std().detach())
            # global_depth_loss = loss_utils.margin_l2_loss(nrpwg, nmpwg, 0.0002, False)

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss_depth_pseudo = loss_scale * args.W * (depth_loss_pseudo + 0.5 * patch_depth_loss_pseudo)
                loss += loss_depth_pseudo

            # ---------------------- crop dino psoude ----------------------
            rendered_image_pseudo = render_pkg_pseudo["render"]
            gt_crop, rendered_crop = loss_utils.random_crop1(gt_image, rendered_image_pseudo, (84, 63))  # for llff dataset
            # gt_crop, rendered_crop = loss_utils.random_crop3(gt_image, rendered_image_pseudo ,(83,63))  # for mipnerf 360 dataset
            render_crop_pseudo_vit = get_vit_feature(rendered_crop.unsqueeze(0), vit_ext0)
            gt_crop_vit = get_vit_feature(gt_crop.unsqueeze(0), vit_ext0)
            side_loss_dino = F.mse_loss(render_crop_pseudo_vit, gt_crop_vit)
            loss_dino = args.D * side_loss_dino
            loss += loss_dino

            flag_pseudo = True

        loss.backward()

        torch.cuda.empty_cache()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "points": f"{scene.gaussians.get_xyz.shape[0]}",
                    "L1": f"{loss_l1_ssim:.4f}",
                    "depth": f"{loss_depth:.4f}",
                    "d_pse": f"{loss_depth_pseudo:.4f}" if flag_pseudo else "X.XXXX",
                    "dino": f"{loss_dino:.4f}" if flag_pseudo else "X.XXXX",
                })

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background), metrics)

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent,
                                                size_threshold, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene: Scene, renderFunc, renderArgs,
                    metrics):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if config['name'] == 'test':
                    metrics['PSNR'].append(float(f"{psnr_test.item():.4f}"))
                    metrics['SSIM'].append(float(f"{ssim_test.item():.4f}"))
                    metrics['LPIPS'].append(float(f"{lpips_test.item():.4f}"))
                    # test_result.append(float(f"{psnr_test.item():.4f}"))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        # print('------------test_result:',test_result)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[10_00, 20_00, 30_00, 40_00, 50_00, 60_00, 70_00, 80_00, 90_00, 10_000, 11_000, 12_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--W", type=float, default=0.5)
    parser.add_argument("--D", type=float, default=0.8)
    parser.add_argument("--N", type=float)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    scene_data_dir = lp.extract(args).source_path
    scene = scene_data_dir.split('/')[-1]
    opt_dir = args.model_path.split('/')[0]

    print(args.test_iterations)

    print("Optimizing " + args.model_path)
    metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, metrics)

    print("\nTraining complete.")
    
    # Saving scores
    import json
    output_dir = os.path.dirname(args.model_path)
    output_data = {"PSNR": metrics['PSNR'], "SSIM": metrics['SSIM'], "LPIPS": metrics['LPIPS'], "scene": scene}
    output_path = os.path.join(output_dir, "test_results.json")
    with open(output_path, "a+") as f:
        json.dump(output_data, f, indent=4)