from copy import deepcopy

from typing import List, Union

from camera_control.Module.camera import Camera

from fast_gs.Model.gs import GaussianModel
from fast_gs.Module.gs_renderer import GSRenderer


def render_cameras(
    gaussians: Union[str, GaussianModel],
    camera_list: List[Camera],
    sh_degree: int = 3,
    bg_color: list = [1, 1, 1],
    mult: float = 0.5,
    device: str = 'cuda:0',
) -> List[Camera]:
    """Render FastGS images + depths into a fresh copy of the cameras.

    输入的 ``camera_list`` 不会被修改：函数内部先 ``deepcopy`` 出一份
    ``rendered_camera_list``，把每个相机的渲染图、深度写回该副本并把
    ``mask`` 置空后返回。该接口与 ``twod_gs.API.renderer.render_cameras``
    保持同签名、同语义（``mult`` 为 FastGS 专有参数）。
    """
    rendered_camera_list = deepcopy(camera_list)

    render_list = GSRenderer.renderCameras(
        gaussians,
        rendered_camera_list,
        sh_degree=sh_degree,
        bg_color=bg_color,
        mult=mult,
        device=device,
        render_depth=True,
    )

    for i, rendered_camera in enumerate(rendered_camera_list):
        image = render_list[i]['render'].permute(1, 2, 0)
        depth = render_list[i]['depth']

        rendered_camera.loadImage(image)
        rendered_camera.loadDepth(depth)

        rendered_camera.mask = None

    return rendered_camera_list
