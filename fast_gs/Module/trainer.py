import os
import sys
sys.path.append('../../../base-gs-trainer/')
import contextlib
import torch

from torch import nn
from tqdm import tqdm
from typing import Tuple

from fused_ssim import fused_ssim

from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from fast_gs.Config.config import ModelParams, PipelineParams, OptimizationParams
from fast_gs.Model.gs import GaussianModel
from fast_gs.Method.render_kernel import render_fastgs
from fast_gs.Method.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


def _log_autograd_state(tag: str) -> None:
    """Print thread-local autograd / inference-mode state.

    Mirrors the probe in ``tools.shape_kernel._log_autograd_state`` /
    ``tools.segment_kernel._log_autograd_state`` so logs from the full
    pipeline (SAM3 -> shape_kernel.fit_fastgs -> FastGSTrainer) can be
    correlated when debugging the ``element 0 of tensors does not
    require grad and does not have a grad_fn`` failure.
    """
    try:
        grad_enabled = torch.is_grad_enabled()
    except Exception:
        grad_enabled = 'unknown'
    inference_enabled = 'unknown'
    if hasattr(torch, 'is_inference_mode_enabled'):
        try:
            inference_enabled = torch.is_inference_mode_enabled()
        except Exception:
            pass
    print(
        f'[autograd-probe][{tag}] '
        f'grad_enabled={grad_enabled} '
        f'inference_mode_enabled={inference_enabled}'
    )


@contextlib.contextmanager
def _force_autograd_on(tag: str):
    """Force autograd on regardless of the caller's thread state.

    Matches ``tools.shape_kernel._force_autograd_on``. Putting the same
    guard inside the trainer itself keeps FastGS robust to a leaked
    ``torch.inference_mode`` / disabled ``grad_enabled`` state from a
    previous serial stage (e.g. SAM3's
    ``@torch.inference_mode()``-decorated streaming predictor), so the
    trainer no longer depends on the external ``fit_fastgs`` wrapper
    doing this for it. Both managers are idempotent when the thread
    is already in the normal training state, so stacking them is safe.
    """
    _log_autograd_state(f'{tag}:before_guard')
    inference_cm = None
    if hasattr(torch, 'inference_mode'):
        try:
            inference_cm = torch.inference_mode(False)
        except TypeError:
            inference_cm = None
    with contextlib.ExitStack() as enter_ctx:
        if inference_cm is not None:
            enter_ctx.enter_context(inference_cm)
        enter_ctx.enter_context(torch.enable_grad())
        _log_autograd_state(f'{tag}:inside_guard')
        yield
    _log_autograd_state(f'{tag}:after_guard')


class Trainer(BaseGSTrainer):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=10000,
        save_freq: int=10000,
    ) -> None:
        self.dataset = ModelParams.default()
        self.opt = OptimizationParams.default()
        self.pipe = PipelineParams.default()

        self.dataset.source_path = os.path.abspath(colmap_data_folder_path)
        self.dataset.model_path = save_result_folder_path

        print("Optimizing " + self.dataset.model_path)

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
        return

    def renderImage(self, viewpoint_cam) -> dict:
        return render_fastgs(viewpoint_cam, self.gaussians, self.pipe, self.background, self.opt.mult)

    def _assert_backward_ready(
        self,
        iteration: int,
        image: torch.Tensor,
        rgb_loss: torch.Tensor,
        ssim_loss: torch.Tensor,
        opacity_loss: torch.Tensor,
        scaling_loss: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> None:
        """Fail fast with a targeted error when ``total_loss`` has no graph.

        The native PyTorch message for this situation is:
            ``element 0 of tensors does not require grad and does not
            have a grad_fn``
        which is ambiguous between two very different root causes:

        1. The caller left the thread in ``torch.inference_mode()`` /
           ``set_grad_enabled(False)``, so the whole forward ran without
           building a graph.
        2. A specific forward node (e.g. a custom CUDA rasterizer or
           ``fused_ssim``) returned a tensor with no ``grad_fn`` even
           though autograd was enabled.

        We check both up front and raise ``RuntimeError`` with the exact
        distinction, including thread-local autograd state and the
        ``requires_grad`` / ``grad_fn`` of every loss component, so the
        web worker vs CLI divergence can be diagnosed from a single log
        line.
        """
        if total_loss.requires_grad and total_loss.grad_fn is not None:
            return

        try:
            grad_enabled = torch.is_grad_enabled()
        except Exception:
            grad_enabled = None
        inference_enabled = None
        if hasattr(torch, 'is_inference_mode_enabled'):
            try:
                inference_enabled = torch.is_inference_mode_enabled()
            except Exception:
                pass

        def _desc(name: str, t: torch.Tensor) -> str:
            return (
                f'{name}(requires_grad={bool(t.requires_grad)}, '
                f'grad_fn={t.grad_fn!r})'
            )

        thread_state = (
            f'grad_enabled={grad_enabled} '
            f'inference_mode_enabled={inference_enabled}'
        )
        tensor_state = ', '.join([
            _desc('image', image),
            _desc('rgb_loss', rgb_loss),
            _desc('ssim_loss', ssim_loss),
            _desc('opacity_loss', opacity_loss),
            _desc('scaling_loss', scaling_loss),
            _desc('total_loss', total_loss),
        ])

        if grad_enabled is False or inference_enabled is True:
            root_cause = (
                'caller left the thread in a no-grad / inference_mode '
                'state; re-enable autograd before calling train()'
            )
        elif not image.requires_grad or image.grad_fn is None:
            root_cause = (
                'forward render produced a tensor with no grad_fn; '
                'inspect render_fastgs / diff_gaussian_rasterization_fastgs'
            )
        else:
            root_cause = (
                'total_loss lost its graph even though image has a '
                'grad_fn; inspect per-component loss tensors above'
            )

        raise RuntimeError(
            f'[fastgs-trainer] iteration={iteration} cannot backward: '
            f'{root_cause}. thread_state=({thread_state}); '
            f'tensor_state=({tensor_state})'
        )

    def trainStep(
        self,
        iteration: int,
        viewpoint_cam,
        lambda_dssim: float = 0.2,
        lambda_opacity: float = 1e-3,
        lambda_scaling: float = 10.0,
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
            opacity_loss = lambda_opacity * nn.L1Loss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0 and iteration < self.opt.densify_until_iter:
            scaling_loss = lambda_scaling * nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        total_loss = \
            (1.0 - lambda_dssim) * rgb_loss + \
            lambda_dssim * ssim_loss + \
            opacity_loss + \
            scaling_loss

        self._assert_backward_ready(
            iteration=iteration,
            image=image,
            rgb_loss=rgb_loss,
            ssim_loss=ssim_loss,
            opacity_loss=opacity_loss,
            scaling_loss=scaling_loss,
            total_loss=total_loss,
        )

        total_loss.backward()

        loss_dict = {
            'rgb': rgb_loss.item(),
            'ssim': ssim_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
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
        return True

    def train(self, iteration_num: int = 30000):
        # Training-level autograd guard. Even though ``fit_fastgs`` (and
        # any other call site) is expected to enter ``_force_autograd_on``
        # around the trainer, we repeat the guard here so the trainer is
        # self-contained: it works identically from the CLI pipeline,
        # the long-lived web worker, and stand-alone scripts, regardless
        # of what torch.inference_mode / set_grad_enabled state a
        # previous stage may have leaked onto this thread.
        with _force_autograd_on('fastgs-trainer.train'):
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

                if iteration % self.test_freq == 0:
                    self.logImageStep(
                        iteration,
                        render_image_num=1,
                        is_fast=True,
                    )

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
