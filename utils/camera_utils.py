import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal


def _load_cam_preprocess(args, id, cam_info):
    """仅在 CPU 上做图像与分辨率预处理，不创建 Camera、不触碰 CUDA。供多线程调用。"""
    image_rgb = PILtoTorch(cam_info.image)

    return {
        "id": id,
        "colmap_id": cam_info.uid,
        "R": cam_info.R,
        "T": cam_info.T,
        "FoVx": cam_info.FovX,
        "FoVy": cam_info.FovY,
        "gt_image": image_rgb,
        "gt_alpha_mask": None,
        "image_name": cam_info.image_name,
        "data_device": args.data_device,
    }

def loadCam(args, id, cam_info):
    pre = _load_cam_preprocess(args, id, cam_info)
    return Camera(
        colmap_id=pre["colmap_id"], R=pre["R"], T=pre["T"],
        FoVx=pre["FoVx"], FoVy=pre["FoVy"],
        image=pre["gt_image"], gt_alpha_mask=pre["gt_alpha_mask"],
        image_name=pre["image_name"], uid=pre["id"], data_device=pre["data_device"],
    )


def cameraList_from_camInfos(cam_infos, args):
    items = list(enumerate(cam_infos))
    # 阶段一：多线程仅在 CPU 上预处理图像，不触碰 CUDA
    with ThreadPoolExecutor() as executor:
        preprocessed = list(tqdm(
            executor.map(
                lambda item: _load_cam_preprocess(args, item[0], item[1]),
                items,
            ),
            total=len(items),
            desc="Loading cameras (preprocess)",
        ))
    # 阶段二：主线程顺序创建 Camera（所有 .cuda() 在此执行，避免 lazy wrapper）
    camera_list = [
        Camera(
            colmap_id=p["colmap_id"], R=p["R"], T=p["T"],
            FoVx=p["FoVx"], FoVy=p["FoVy"],
            image=p["gt_image"], gt_alpha_mask=p["gt_alpha_mask"],
            image_name=p["image_name"], uid=p["id"], data_device=p["data_device"],
        )
        for p in tqdm(preprocessed, desc="Loading cameras (to GPU)")
    ]
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
