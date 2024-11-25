import argparse
import os
import random
from functools import partial

import imageio
import numpy as np
import torch
import torch.distributed
import torchvision.transforms.functional as F
from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from einops import rearrange
from omegaconf import OmegaConf
from sgm.util import get_obj_from_str, isheatmap
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image
from utils.traj_utils import bivariate_Gaussian

from sat import mpu
from sat.training.deepspeed_training import training_main


try:
    import wandb
except ImportError:
    print("warning: wandb not installed")


def print_debug(args, s):
    if args.debug:
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        print(s)


def save_texts(texts, save_dir, iterations):
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)
        if args is not None and args.wandb:
            wandb.log(
                {key + f"_video_{i}": wandb.Video(now_save_path, fps=fps, format="mp4")}, step=args.iteration + 1
            )


def preprocess(img1_batch, img2_batch, transforms, reshape_size):
    img1_batch = F.resize(img1_batch, size=reshape_size, antialias=False)
    img2_batch = F.resize(img2_batch, size=reshape_size, antialias=False)
    return transforms(img1_batch, img2_batch)


@torch.no_grad()
def get_raft_flow(raft_model, frames, reshape_size, raft_batchsize):
    device = list(raft_model.parameters())[0].device
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    current_frames, next_frames = preprocess(frames[:-1], frames[1:], transforms, reshape_size)
    current_frames, next_frames = current_frames.to(device), next_frames.to(device)
    predicted_flows = []
    for i in range(0, len(current_frames), raft_batchsize):
        current_frame, next_frame = current_frames[i : i + raft_batchsize], next_frames[i : i + raft_batchsize]
        list_of_flows = raft_model(current_frame, next_frame)
        predicted_flows.append(list_of_flows[-1])
        del list_of_flows
    predicted_flows = torch.cat(predicted_flows)
    return predicted_flows


@torch.no_grad()
def get_keypoint(first_flow, lamb, reshape_size, max_sample_num):
    H, W = reshape_size
    offset_h, offset_w = random.randint(0, lamb - 1), random.randint(0, lamb - 1)
    h_indice = torch.tensor(range(offset_h, H, lamb), dtype=torch.int)
    w_indice = torch.tensor(range(offset_w, W, lamb), dtype=torch.int)
    new_w = w_indice.shape[0]
    sample_flow = first_flow[..., offset_h::lamb, offset_w::lamb]
    sample_flow = rearrange(sample_flow, "c h w  -> c (h w)")
    candidates_val = torch.norm(sample_flow.to(float), dim=0, p=2)
    total_flow_val = torch.sum(candidates_val)
    candidates_val = candidates_val / total_flow_val
    sample_N = random.randint(1, max_sample_num)
    sample_list = np.random.choice(
        range(len(candidates_val)), size=sample_N, p=candidates_val.detach().cpu().numpy(), replace=False
    )
    point_list = [[[w_indice[idx % new_w].item(), h_indice[idx // new_w].item()]] for idx in sample_list]
    return point_list


@torch.no_grad()
def get_trajs(
    raft_model,
    clip,
    raft_batchsize,
    video_len,
    reshape_size=(480, 480),
    lamb=32,
    max_sample_num=16,
    sample=False,
):
    original_flow = get_raft_flow(raft_model, clip, reshape_size, raft_batchsize)  # [T, 2, reshape_H, reshape_W]
    H, W = reshape_size
    size, sigma = 99, 10
    blur_kernel = bivariate_Gaussian(size, sigma, sigma, 0, grid=None, isotropic=True)
    blur_kernel = blur_kernel / blur_kernel[size // 2, size // 2]

    # do sparse tracklets sampling
    if sample:
        flow = original_flow[0]
        point_list = get_keypoint(flow, lamb, reshape_size, max_sample_num)

        sample_flow = torch.zeros((video_len, 2, H, W), device=original_flow.device)
        for t in range(1, video_len):
            for n in range(len(point_list)):
                prev_x, prev_y = point_list[n][-1]
                cur_x, cur_y = (prev_x + int(flow[0, prev_y, prev_x])), (prev_y + int(flow[1, prev_y, prev_x]))
                cur_x = max(min(W - 1, cur_x), 0)
                cur_y = max(min(H - 1, cur_y), 0)
                sample_flow[t, 0, prev_y, prev_x] = cur_x - prev_x
                sample_flow[t, 1, prev_y, prev_x] = cur_y - prev_y
                point_list[n].append([cur_x, cur_y])
            if t != video_len - 1:
                flow = original_flow[t]

        blur_kernel = torch.from_numpy(blur_kernel).to(original_flow.device, torch.float32)[None, None]
        sample_flow = rearrange(sample_flow, "t c h w -> (t c) 1 h w")
        sample_flow = torch.nn.functional.conv2d(sample_flow, blur_kernel, padding="same")
        original_flow = rearrange(sample_flow, "(t c) 1 h w -> t h w c", c=2)
    else:
        original_flow = torch.cat([torch.zeros(1, 2, H, W).to(original_flow.device), original_flow])
        original_flow = original_flow.permute(0, 2, 3, 1)

    return original_flow


@torch.no_grad()
def get_flow_from_points(
    point_list,
    device="cpu",
    reshape_size=(480, 480),
):
    video_len = len(point_list)
    H, W = reshape_size
    size, sigma = 99, 10
    blur_kernel = bivariate_Gaussian(size, sigma, sigma, 0, grid=None, isotropic=True)
    blur_kernel = blur_kernel / blur_kernel[size // 2, size // 2]

    num_objects = len(point_list[0])
    sample_flow = torch.zeros((video_len, 2, H, W), device=device)
    for t in range(1, video_len):
        for n in range(num_objects):
            prev = point_list[t - 1][n]
            cur = point_list[t][n]
            if prev is None or cur is None:
                continue
            prev_x, prev_y = prev
            cur_x, cur_y = cur
            if 0 <= prev_x < W and 0 <= prev_y < H:
                sample_flow[t, 0, prev_y, prev_x] = cur_x - prev_x
                sample_flow[t, 1, prev_y, prev_x] = cur_y - prev_y

    blur_kernel = torch.from_numpy(blur_kernel).to(device, torch.float32)[None, None]
    sample_flow = rearrange(sample_flow, "t c h w -> (t c) 1 h w")
    sample_flow = torch.nn.functional.conv2d(sample_flow, blur_kernel, padding="same")
    original_flow = rearrange(sample_flow, "(t c) 1 h w -> t h w c", c=2)

    return original_flow


def flow_to_image_in_batches(flow, batch_size):
    res = []
    for i in range(0, len(flow), batch_size):
        flow_b = flow[i : i + batch_size]
        res.append(flow_to_image(flow_b))
    return torch.cat(res)


def log_video(batch, model, args, only_log_video_latents=False):
    texts = batch["txt"]
    text_save_dir = os.path.join(args.save, "video_texts")
    os.makedirs(text_save_dir, exist_ok=True)
    save_texts(texts, text_save_dir, args.iteration)

    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)

    if torch.distributed.get_rank() == 0:
        root = os.path.join(args.save, "video")

        if only_log_video_latents:
            root = os.path.join(root, "latents")
            filename = "{}_gs-{:06}".format("latents", args.iteration)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            os.makedirs(path, exist_ok=True)
            torch.save(videos["latents"], os.path.join(path, "latent.pt"))
        else:
            for k in videos:
                N = videos[k].shape[0]
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()
                    if not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)

            num_frames = batch["num_frames"][0]
            fps = batch["fps"][0].cpu().item()
            if only_log_video_latents:
                root = os.path.join(root, "latents")
                filename = "{}_gs-{:06}".format("latents", args.iteration)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                os.makedirs(path, exist_ok=True)
                torch.save(videos["latents"], os.path.join(path, "latents.pt"))
            else:
                for k in videos:
                    samples = (videos[k] + 1.0) / 2.0
                    filename = "{}_gs-{:06}".format(k, args.iteration)

                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    save_video_as_grid_and_mp4(samples, path, num_frames // fps, fps, args, k)


def broad_cast_batch(batch):
    mp_size = mpu.get_model_parallel_world_size()
    global_rank = torch.distributed.get_rank() // mp_size
    src = global_rank * mp_size

    if batch["mp4"] is not None:
        broadcast_shape = [batch["mp4"].shape, batch["fps"].shape, batch["num_frames"].shape]
    else:
        broadcast_shape = None

    objects_to_broadcast = [batch["txt"], broadcast_shape, batch["points"]]
    torch.distributed.broadcast_object_list(objects_to_broadcast, src=src, group=mpu.get_model_parallel_group())
    batch["txt"] = objects_to_broadcast[0]
    mp4_shape, fps_shape, num_frames_shape = objects_to_broadcast[1]
    batch["points"] = objects_to_broadcast[2]

    if mpu.get_model_parallel_rank() != 0:
        batch["mp4"] = torch.zeros(mp4_shape, device="cuda")
        batch["fps"] = torch.zeros(fps_shape, device="cuda", dtype=torch.long)
        batch["num_frames"] = torch.zeros(num_frames_shape, device="cuda", dtype=torch.long)

    torch.distributed.broadcast(batch["mp4"], src=src, group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(batch["fps"], src=src, group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(batch["num_frames"], src=src, group=mpu.get_model_parallel_group())
    return batch


@torch.no_grad()
def get_training_flow(args, video_latent, device, dtype, points=None):
    """
    args:
        video_latent: [B T C H W] shape Tensor.

    out:
        video_flows: [B C T H W] shape Tensor. data range: [-1, 1]
    """
    if args.use_raft:
        raft_model.to(device)
    video_flows = []
    B, T, C, H, W = video_latent.shape
    raft_batchsize = max(video_latent.shape[0] * video_latent.shape[1] // 20, 1)  # 20 is empirical.
    video_for_raft = (video_latent + 1) / 2
    for b in range(video_for_raft.shape[0]):
        if points is not None and not args.use_raft:
            video_flow = get_flow_from_points(points[b], device=device, reshape_size=(H, W))
        else:
            video_flow = get_trajs(
                raft_model,
                video_for_raft[b],
                raft_batchsize=raft_batchsize,
                video_len=T,
                reshape_size=(video_for_raft.shape[-2:]),
                sample=args.sample_flow,
            )
        video_flow = flow_to_image_in_batches(rearrange(video_flow, "T H W C -> T C H W"), raft_batchsize)
        video_flows.append(video_flow)
    video_flows = torch.stack(video_flows)
    if args.vis_traj_features:
        os.makedirs("samples/flow", exist_ok=True)
        imageio.mimwrite(
            "samples/flow/flow2img.gif",
            rearrange(video_flows.cpu(), "B T C H W -> (B T) H W C"),
            fps=8,
            loop=0,
        )
    video_flows = rearrange(video_flows / 255.0 * 2 - 1, "B T C H W -> B C T H W")
    video_flows = video_flows.to(device, dtype)
    return video_flows


def forward_step_eval(data_iterator, model, args, timers, only_log_video_latents=False, data_class=None):
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()
        batch_video = next(data_iterator)
        timers("data loader").stop()

        if len(batch_video["mp4"].shape) == 6:
            b, v = batch_video["mp4"].shape[:2]
            batch_video["mp4"] = batch_video["mp4"].view(-1, *batch_video["mp4"].shape[2:])
            txt = []
            for i in range(b):
                for j in range(v):
                    txt.append(batch_video["txt"][j][i])
            batch_video["txt"] = txt

        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()
    else:
        batch_video = {"mp4": None, "fps": None, "num_frames": None, "txt": None, "points": None}
    broad_cast_batch(batch_video)
    video_flow = get_training_flow(
        args, batch_video["mp4"], batch_video["mp4"].device, batch_video["mp4"].dtype, points=None
    )
    if mpu.get_data_parallel_rank() == 0:
        log_video(batch_video, model, args, only_log_video_latents=only_log_video_latents)
    batch_video["video_flow"] = video_flow
    batch_video["global_step"] = args.iteration
    loss, loss_dict = model.shared_step(batch_video)
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)
    return loss, loss_dict


def forward_step(data_iterator, model, args, timers, data_class=None):
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()
        batch = next(data_iterator)
        timers("data loader").stop()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        if torch.distributed.get_rank() == 0:
            if not os.path.exists(os.path.join(args.save, "training_config.yaml")):
                configs = [OmegaConf.load(cfg) for cfg in args.base]
                config = OmegaConf.merge(*configs)
                os.makedirs(args.save, exist_ok=True)
                OmegaConf.save(config=config, f=os.path.join(args.save, "training_config.yaml"))

    else:
        batch = {"mp4": None, "fps": None, "num_frames": None, "txt": None, "points": None}

    batch["global_step"] = args.iteration
    if args.model_parallel_size > 1:
        broad_cast_batch(batch)

    video_flow = get_training_flow(
        args, batch["mp4"], batch["mp4"].device, batch["mp4"].dtype, points=batch.get("points")
    )
    batch["video_flow"] = video_flow.contiguous()
    loss, loss_dict = model.shared_step(batch)

    return loss, loss_dict


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    data_class = get_obj_from_str(args.data_config["target"])
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

    import yaml

    configs = []
    for config in args.base:
        with open(config, "r") as f:
            base_config = yaml.safe_load(f)
        configs.append(base_config)
    args.log_config = configs

    if args.use_raft:
        raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(
            int(os.environ.get("LOCAL_RANK", 0))
        )
        raft_model = raft_model.eval()

    training_main(
        args,
        model_cls=SATVideoDiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class),
        forward_step_eval=partial(
            forward_step_eval, data_class=data_class, only_log_video_latents=args.only_log_video_latents
        ),
        create_dataset_function=create_dataset_function,
    )
