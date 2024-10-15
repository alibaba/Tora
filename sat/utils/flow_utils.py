import cv2
import numpy as np
import torch
from utils.traj_utils import bivariate_Gaussian


def read_points(file, video_len=16, reverse=False):
    with open(file, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.strip().split(",")
        points.append((int(x), int(y)))
    if reverse:
        points = points[::-1]

    if len(points) > video_len:
        skip = len(points) // video_len
        points = points[::skip]
    points = points[:video_len]

    return points


size = 99
sigma = 10
blur_kernel = bivariate_Gaussian(size, sigma, sigma, 0, grid=None, isotropic=True)
blur_kernel = blur_kernel / blur_kernel[size // 2, size // 2]


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def process_points(points, frames):
    defualt_points = [[512, 512]] * frames

    if len(points) < 2:
        return defualt_points
    elif len(points) >= frames:
        skip = len(points) // frames
        return points[::skip][: frames - 1] + points[-1:]
    else:
        insert_num = frames - len(points)
        insert_num_dict = {}
        interval = len(points) - 1
        n = insert_num // interval
        m = insert_num % interval
        for i in range(interval):
            insert_num_dict[i] = n
        for i in range(m):
            insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            delta_x = x1 - x0
            delta_y = y1 - y0
            for j in range(insert_num_dict[i]):
                x = x0 + (j + 1) / (insert_num_dict[i] + 1) * delta_x
                y = y0 + (j + 1) / (insert_num_dict[i] + 1) * delta_y
                insert_points.append([int(x), int(y)])

            res += points[i : i + 1] + insert_points
        res += points[-1:]
        return res


def get_flow(points, optical_flow, video_len):
    for i in range(video_len - 1):
        p = points[i]
        p1 = points[i + 1]
        optical_flow[i + 1, p[1], p[0], 0] = p1[0] - p[0]
        optical_flow[i + 1, p[1], p[0], 1] = p1[1] - p[1]

    return optical_flow


def process_traj(points_files, num_frames, video_size, device="cpu"):
    optical_flow = np.zeros((num_frames, video_size[0], video_size[1], 2), dtype=np.float32)
    processed_points = []
    for points_file in points_files:
        points = read_points(points_file, video_len=num_frames)
        xy_range = 256
        h, w = video_size
        points = process_points(points, num_frames)
        points = [[int(w * x / xy_range), int(h * y / xy_range)] for x, y in points]
        optical_flow = get_flow(points, optical_flow, video_len=num_frames)
        processed_points.append(points)

    for i in range(1, num_frames):
        optical_flow[i] = cv2.filter2D(optical_flow[i], -1, blur_kernel)

    optical_flow = torch.tensor(optical_flow).to(device)

    return optical_flow, processed_points


if __name__ == "__main__":
    # points_file = ["assets/trajs/shake_1.txt"]
    points_file = [
        "assets/trajs/outputs/x/00.txt",
        "assets/trajs/outputs/x/01.txt",
        "assets/trajs/outputs/x/02.txt",
        "assets/trajs/outputs/x/03.txt",
    ]
    num_frames = 10
    # video_size = [1216, 720]  # H, W
    # video_size = [848, 480]  # H, W
    # video_size = [480, 848]  # H, W
    video_size = [720, 1280]  # H, W
    # video_size = [720, 720]  # H, W
    # video_size = [1280, 720]  # H, W
    # video_size = [736, 720]  # H, W
    # video_size = [576, 985]  # H, W
    device = "cpu"
    flow, points = process_traj(points_file, num_frames, video_size, device)
    print(flow.shape)
    print(points)
    import pickle

    with open("assets/processed_points/1-1-1.pkl", "wb") as f:
        pickle.dump(points, f)
