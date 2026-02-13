import os
import sys
sys.path.append('../../../base-gs-trainer/')
import torch

from torch import nn
from tqdm import tqdm
from typing import Tuple
from argparse import ArgumentParser

from fused_ssim import fused_ssim

from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Loss.chamfer import chamferLossFn
from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from flexi_cubes.Module.fc_convertor import FCConvertor

from mv_fc_recon.Loss.mesh_geo_energy import thin_plate_energy

from fast_gs.Config.config import ModelParams, PipelineParams, OptimizationParams
from fast_gs.Model.gs import GaussianModel
from fast_gs.Method.render_kernel import render_fastgs
from fast_gs.Method.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


class FCTrainer(BaseGSTrainer):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        init_mesh_file_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=10000,
        save_freq: int=10000,
        fc_update_freq: int=100,
    ) -> None:
        self.fc_update_freq = fc_update_freq

        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(sys.argv[1:])

        args.source_path = colmap_data_folder_path
        args.model_path = save_result_folder_path

        print("Optimizing " + args.model_path)

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)

        BaseGSTrainer.__init__(
            self,
            colmap_data_folder_path=colmap_data_folder_path,
            device=device,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            test_freq=test_freq,
            save_freq=save_freq,
        )

        assert os.path.exists(init_mesh_file_path)
        self.fc_params = FCConvertor.createFC(
            init_mesh_file_path,
            resolution=192,
            device=self.device,
        )

        lr_sdf  = 0.01
        lr_deform = 0.01
        lr_weight = 0.01
        param_groups = [
            dict(params=[self.fc_params['sdf']], lr=lr_sdf),
            dict(params=[self.fc_params['deform']], lr=lr_deform),
            dict(params=[self.fc_params['weight']], lr=lr_weight),
        ]

        self.fc_optimizer = torch.optim.Adam(param_groups)

        self.chamfer_func = chamferLossFn(self.device)

        self.E_thinplate_base = None
        return

    def renderImage(self, viewpoint_cam) -> dict:
        return render_fastgs(viewpoint_cam, self.gaussians, self.pipe, self.background, self.opt.mult)

    def extractMesh(self):
        current_mesh, vertices, L_dev = FCConvertor.extractMesh(self.fc_params, training=True)
        return current_mesh, vertices, L_dev

    def trainStep(
        self,
        iteration: int,
        viewpoint_cam,
        lambda_dssim: float = 0.2,
        lambda_opacity: float = 1e-3,
        lambda_scaling: float = 10.0,
        lambda_chamfer: float = 1.0,
        lambda_dev: float = 0.1,
        lambda_thin_plate: float = 0.1,
    ) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        render_pkg = self.renderImage(viewpoint_cam)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        rgb_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_opacity > 0 and iteration < self.opt.densify_until_iter:
            opacity_loss = nn.L1Loss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0 and iteration < self.opt.densify_until_iter:
            scaling_loss = nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        # FlexiCubes developability 正则化损失
        dev_loss = torch.tensor(0.0, device=self.device)
        chamfer_loss = torch.tensor(0.0, device=self.device)
        if iteration % self.fc_update_freq == 0:
            fc_mesh, vertices, L_dev = self.extractMesh()

            if iteration == 1:
                fc_mesh.export(self.save_result_folder_path + 'start_fc_mesh.ply')

            '''
            fc_depth = NVDiffRastRenderer.renderDepth(
                fc_mesh,
                viewpoint_cam._cam,
                vertices_tensor=vertices,
            )['depth']
            '''

            if L_dev is not None and L_dev.numel() > 0:
                dev_loss = L_dev.mean()

            dists1, dists2 = self.chamfer_func(
                vertices.unsqueeze(0),
                self.gaussians.get_xyz.unsqueeze(0),
            )[:2]

            chamfer_loss = dists1.mean() + dists2.mean()

            '''
            faces = torch.from_numpy(fc_mesh.faces).long().to(self.device)
            if self.E_thinplate_base is None:
                with torch.no_grad():
                    V0 = torch.from_numpy(fc_mesh.vertices).float().to(self.device)
                    self.E_thinplate_base = thin_plate_energy(V0, faces)

                fc_mesh.export(self.save_result_folder_path + 'start_fc_mesh.ply')

            thinplate_loss = thin_plate_energy(vertices, faces, factor=self.E_thinplate_base)
            '''

        total_loss = \
            (1.0 - lambda_dssim) * rgb_loss + \
            lambda_dssim * ssim_loss + \
            lambda_opacity * opacity_loss + \
            lambda_scaling * scaling_loss + \
            lambda_chamfer * chamfer_loss + \
            lambda_dev * dev_loss# + \
            #lambda_thin_plate * thinplate_loss

        total_loss.backward()

        if iteration % 100 == 0:
            self.fc_optimizer.step()
            self.fc_optimizer.zero_grad()

        loss_dict = {
            'rgb': rgb_loss.item(),
            'ssim': ssim_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
            'chamfer': chamfer_loss.item(),
            'dev': dev_loss.item(),
            #'thinplate': thinplate_loss.item(),
            'total': total_loss.item(),
        }

        return render_pkg, loss_dict

    @torch.no_grad()
    def recordGrads(self, render_pkg: dict) -> bool:
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        return True

    @torch.no_grad()
    def densifyStep(self, render_pkg: dict) -> bool:
        size_threshold = 20
        my_viewpoint_stack = self.scene.train_cameras
        camlist = sampling_cameras(my_viewpoint_stack)

        # The multiview consistent densification of fastgs
        importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, self.gaussians, self.pipe, self.background, self.opt, DENSIFY=True)
        self.gaussians.densify_and_prune_fastgs(
            max_screen_size = size_threshold,
            min_opacity = 0.005,
            extent = self.scene.cameras_extent,
            radii=render_pkg['radii'],
            args = self.opt,
            importance_score = importance_score,
            pruning_score = pruning_score,
        )
        return True

    @torch.no_grad()
    def resetOpacity(self) -> bool:
        self.gaussians.reset_opacity()
        return True

    @torch.no_grad()
    def finalPrune(self) -> bool:
        my_viewpoint_stack = self.scene.train_cameras
        camlist = sampling_cameras(my_viewpoint_stack)

        _, pruning_score = compute_gaussian_score_fastgs(camlist, self.gaussians, self.pipe, self.background, self.opt)
        self.gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
        return True

    @torch.no_grad()
    def updateGSParams(self, iteration: int) -> bool:
        self.gaussians.optimizer_step(iteration)
        return True

    @torch.no_grad()
    def saveScene(self, iteration: int) -> bool:
        point_cloud_path = os.path.join(self.dataset.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        fc_mesh = self.extractMesh()[0]
        fc_mesh.export(os.path.join(point_cloud_path, 'fc_mesh.ply'))
        return True

    def train(self, iteration_num: int = 30000):
        progress_bar = tqdm(desc="Training progress", total=iteration_num)
        iteration = 1
        for _ in range(iteration_num):
            viewpoint_cam = self.scene[iteration]

            render_pkg, loss_dict = self.trainStep(iteration, viewpoint_cam)

            if iteration % 10 == 0:
                bar_loss_dict = {
                    "rgb": f"{loss_dict['rgb']:.{5}f}",
                    "ssim": f"{loss_dict['ssim']:.{5}f}",
                    "total": f"{loss_dict['total']:.{5}f}",
                    "Points": f"{len(self.gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(bar_loss_dict)
                progress_bar.update(10)

            self.logStep(iteration, loss_dict)

            if iteration % self.save_freq == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                self.saveScene(iteration)

            # Densification
            if iteration < self.opt.densify_until_iter:
                self.recordGrads(render_pkg)
                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    self.densifyStep(render_pkg)

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.resetOpacity()

            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
            # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
            # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
            if iteration % 3000 == 0 and iteration > self.opt.densify_until_iter:
                self.finalPrune()

            self.updateGSParams(iteration)

            iteration += 1
            self.iteration = iteration
        return True
