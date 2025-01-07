try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import os
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
import random
import json
import uuid
from tqdm import tqdm

from gaussian_utils import (
    loss_utils,
    depth_utils,
    general_utils
)
from gaussian_utils.loss_utils import (
    l1_loss, 
    l1_loss_mask,
    ssim,
    get_vit_feature
)
from gaussian_utils.depth_utils import estimate_depth
from gaussian_utils.image_utils import psnr
from gaussian_utils.extractor import VitExtractor
from gaussian_renderer import render
from scene import Scene, GaussianModel
from lpipsPyTorch import lpips

class GaussianTrainer:
    def __init__(self, dataset, opt, pipe, args):
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.args = args
        self.metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}
        self.tb_writer = self._prepare_output_and_logger()
        self.gaussians = GaussianModel(args)
        self.scene = Scene(args, self.gaussians, shuffle=False)
        self.background = self._setup_background()
        self.vit_extractor = VitExtractor(model_name='dino_vits16', device="cuda:0")
        
    def _setup_background(self):
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        return torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
    def _prepare_output_and_logger(self):
        if not self.dataset.model_path:
            unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4())[0:10])
            self.dataset.model_path = os.path.join("./output/", unique_str)
        
        os.makedirs(self.dataset.model_path, exist_ok=True)
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(self.dataset.model_path)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer
        # return SummaryWriter(self.dataset.model_path)

    def _handle_depth_loss(self, rendered_depth, midas_depth):
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)
        
        depth_loss = min(
            (1 - pearson_corrcoef(-midas_depth, rendered_depth)),
            (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        return self.args.depth_weight * depth_loss

    def _handle_pseudo_sampling(self, iteration, gt_image, viewpoint_cam, pseudo_stack):
        if not self._should_sample_pseudo(iteration):
            return 0, pseudo_stack
            
        if not pseudo_stack:
            pseudo_stack = self.scene.getPseudoCameras().copy()
        
        pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
        render_pkg_pseudo = render(pseudo_cam, self.gaussians, self.pipe, self.background)
        
        loss = self._compute_pseudo_losses(render_pkg_pseudo, gt_image, iteration)
        return loss, pseudo_stack

    def _should_sample_pseudo(self, iteration):
        return (iteration % self.args.sample_pseudo_interval == 0 and 
                self.args.start_sample_pseudo < iteration < self.args.end_sample_pseudo)

    def _compute_pseudo_losses(self, render_pkg_pseudo, gt_image, iteration):
        rendered_depth = render_pkg_pseudo["depth"][0]
        midas_depth = estimate_depth(render_pkg_pseudo["render"], mode='train')
        
        # Window processing
        rdp4 = rendered_depth.unsqueeze(0).unsqueeze(0)
        mdp4 = midas_depth.unsqueeze(0).unsqueeze(0)
        rendered_windows = loss_utils.window_partition3(rdp4, 126, 64)
        midas_windows = loss_utils.window_partition3(mdp4, 126, 64)
        
        # Global depth loss
        depth_loss = self._compute_depth_correlation(rendered_depth, midas_depth)
        
        # Window-based depth loss
        patch_loss = self._compute_window_depth_loss(rendered_windows, midas_windows)
        
        # DINO loss
        dino_loss = self._compute_dino_loss(render_pkg_pseudo["render"], gt_image)
        
        # Combine losses
        loss_scale = min((iteration - self.args.start_sample_pseudo) / 500., 1)
        total_loss = loss_scale * self.args.N * (depth_loss + self.args.W * patch_loss)
        total_loss += self.args.D * dino_loss
        
        return total_loss

    def _compute_depth_correlation(self, rendered_depth, midas_depth):
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)
        return (1 - pearson_corrcoef(rendered_depth, -midas_depth)).mean()

    def _compute_window_depth_loss(self, rendered_windows, midas_windows):
        rendered_windows = rendered_windows.squeeze(1)
        midas_windows = midas_windows.squeeze(1)
        
        rn = random.randint(0, rendered_windows.shape[0] - 1)
        rendered_patch = rendered_windows[rn].reshape(-1, 1)
        midas_patch = midas_windows[rn].reshape(-1, 1)
        
        nrpw0 = loss_utils.normalize0(rendered_patch)
        nmpw0 = loss_utils.normalize0(midas_patch)
        
        return (1 - pearson_corrcoef(nrpw0, -nmpw0)).mean()

    def _compute_dino_loss(self, rendered_image, gt_image):
        gt_crop, rendered_crop = loss_utils.random_crop1(gt_image, rendered_image, (84, 63))
        render_feat = get_vit_feature(rendered_crop.unsqueeze(0), self.vit_extractor)
        gt_feat = get_vit_feature(gt_crop.unsqueeze(0), self.vit_extractor)
        return F.mse_loss(render_feat, gt_feat)

    def _update_progress(self, progress_bar, loss, iteration):
        """Updates the progress bar with current training metrics."""
        progress_bar.set_postfix({
            "points": f"{self.scene.gaussians.get_xyz.shape[0]}",
            "L1": f"{loss.item():.4f}",
            "depth": f"{self._handle_depth_loss(self.render_pkg['depth'][0], self.midas_depth):.4f}",
            "d_pse": f"{self.pseudo_loss:.4f}" if hasattr(self, 'pseudo_loss') else "X.XXXX",
            "dino": f"{self.dino_loss:.4f}" if hasattr(self, 'dino_loss') else "X.XXXX"
        })
        progress_bar.update(10)

    def _save_checkpoint(self, iteration):
        """Saves model checkpoint with current iteration."""
        print(f"\n[ITER {iteration}] Saving Checkpoint")
        torch.save(
            (self.gaussians.capture(), iteration),
            f"{self.scene.model_path}/chkpnt{iteration}.pth"
        )

    def train(self):
        self.gaussians.training_setup(self.opt)
        
        if self.args.start_checkpoint:
            model_params, first_iter = torch.load(self.args.start_checkpoint)
            self.gaussians.restore(model_params, self.opt)
        else:
            first_iter = 0

        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        viewpoint_stack = None
        pseudo_stack = None
        ema_loss_for_log = 0.0

        for iteration in range(first_iter + 1, self.opt.iterations + 1):
            loss = self._training_step(iteration, viewpoint_stack, pseudo_stack, progress_bar)
            
            if iteration % 10 == 0:
                self._update_progress(progress_bar, loss, iteration)

            if iteration in self.args.save_iterations:
                self.scene.save(iteration)

            if iteration in self.args.checkpoint_iterations:
                self._save_checkpoint(iteration)

        self._save_results()

    def _training_step(self, iteration, viewpoint_stack, pseudo_stack, progress_bar):
        if iteration % 500 == 0:
            self.gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = self.scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background)
        self.render_pkg = render_pkg
        self.midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        # Calculate primary losses
        gt_image = viewpoint_cam.original_image.cuda()
        l1_loss = l1_loss_mask(render_pkg["render"], gt_image)
        loss_l1_ssim = (1.0 - self.opt.lambda_dssim) * l1_loss + \
                      self.opt.lambda_dssim * (1.0 - ssim(render_pkg["render"], gt_image))
        
        # Add depth loss
        loss = loss_l1_ssim + self._handle_depth_loss(
            render_pkg["depth"][0], 
            self.midas_depth
        )

        # Handle pseudo sampling if needed
        pseudo_loss, pseudo_stack = self._handle_pseudo_sampling(
            iteration, gt_image, viewpoint_cam, pseudo_stack
        )
        self.pseudo_loss = pseudo_loss
        loss += pseudo_loss

        loss.backward()
        
        self._optimization_step(iteration, render_pkg, viewpoint_stack)
        
        return loss

    def _optimization_step(self, iteration, render_pkg, viewpoint_stack):
        if iteration < self.opt.densify_until_iter:
            visibility_filter = render_pkg["visibility_filter"]
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter],
                render_pkg["radii"][visibility_filter]
            )
            self.gaussians.add_densification_stats(
                render_pkg["viewspace_points"], 
                visibility_filter
            )

            if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                self.gaussians.densify_and_prune(
                    self.opt.densify_grad_threshold,
                    self.opt.prune_threshold,
                    self.scene.cameras_extent,
                    None,
                    iteration
                )

        if iteration < self.opt.iterations:
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.gaussians.update_learning_rate(iteration)

        if (iteration - self.args.start_sample_pseudo - 1) % self.opt.opacity_reset_interval == 0 and \
                iteration > self.args.start_sample_pseudo:
            self.gaussians.reset_opacity()

    def _save_results(self):
        output_dir = os.path.dirname(self.dataset.model_path)
        output_data = {
            "PSNR": self.metrics['PSNR'],
            "SSIM": self.metrics['SSIM'],
            "LPIPS": self.metrics['LPIPS'],
            "scene": self.dataset.source_path.split('/')[-1]
        }
        
        output_path = os.path.join(output_dir, "test_results.json")
        with open(output_path, "a+") as f:
            json.dump(output_data, f, indent=4)

def main():
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams
    
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # Add additional arguments
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                      default=[1000 * i for i in range(1, 11)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--train_bg", action="store_true")
    parser.add_argument("--W", type=float, default=0.5)
    parser.add_argument("--D", type=float, default=0.8)
    parser.add_argument("--N", type=float, default=1.0)
    
    args = parser.parse_args()
    
    general_utils.safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    trainer = GaussianTrainer(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args
    )
    trainer.train()
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
