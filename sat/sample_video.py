import argparse
import gc
import json
import math
import os
import pickle
from pathlib import Path
from typing import List, Union

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as TT
from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from einops import rearrange, repeat
from omegaconf import ListConfig
from PIL import Image
from torchvision.io import write_video
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import tqdm
from utils.flow_utils import process_traj
from utils.misc import vis_tensor

from sat import mpu
from sat.arguments import set_random_seed
from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def draw_points(video, points):
    """
    Draw points onto video frames.

    Parameters:
        video (torch.tensor): Video tensor with shape [T, H, W, C], where T is the number of frames,
                            H is the height, W is the width, and C is the number of channels.
        points (list): Positions of points to be drawn as a tensor with shape [N, T, 2],
                            each point contains x and y coordinates.

    Returns:
        torch.tensor: The video tensor after drawing points, maintaining the same shape [T, H, W, C].
    """

    T = video.shape[0]
    N = len(points)
    device = video.device
    dtype = video.dtype
    video = video.cpu().numpy().copy()
    traj = np.zeros(video.shape[-3:], dtype=np.uint8)  # [H, W, C]
    for n in range(N):
        for t in range(1, T):
            cv2.line(traj, tuple(points[n][t - 1]), tuple(points[n][t]), (255, 1, 1), 2)
    for t in range(T):
        mask = traj[..., -1] > 0
        mask = repeat(mask, "h w -> h w c", c=3)
        alpha = 0.7
        video[t][mask] = video[t][mask] * (1 - alpha) + traj[mask] * alpha
        for n in range(N):
            cv2.circle(video[t], tuple(points[n][t]), 3, (160, 230, 100), -1)
    video = torch.from_numpy(video).to(device, dtype)
    return video


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor,
    save_path: str,
    name: str,
    fps: int = 5,
    args=None,
    key=None,
    traj_points=None,
    prompt="",
):
    os.makedirs(save_path, exist_ok=True)
    p = Path(save_path)

    for i, vid in enumerate(video_batch):
        x = rearrange(vid, "t c h w -> t h w c")
        x = x.mul(255).add(0.5).clamp(0, 255).to("cpu", torch.uint8)  # [T H W C]
        os.makedirs(p / "video", exist_ok=True)
        os.makedirs(p / "prompt", exist_ok=True)
        if traj_points is not None:
            os.makedirs(p / "traj", exist_ok=True)
            os.makedirs(p / "traj_video", exist_ok=True)
            write_video(
                p / "video" / f"{name}_{i:06d}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
            with open(p / "traj" / f"{name}_{i:06d}.pkl", "wb") as f:
                pickle.dump(traj_points, f)
            x = draw_points(x, traj_points)
            write_video(
                p / "traj_video" / f"{name}_{i:06d}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
        else:
            write_video(
                p / "video" / f"{name}_{i:06d}.mp4",
                x,
                fps=fps,
                video_codec="libx264",
                options={"crf": "18"},
            )
        with open(p / "prompt" / f"{name}_{i:06d}.txt", "w") as f:
            f.write(prompt)


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    image_size = [480, 720]

    if args.image2video:
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        transform = TT.Compose(chained_trainsforms)

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device
    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            print(text)
            set_random_seed(args.seed)
            if args.image2video:
                if args.flow_from_prompt:
                    text, image_path, flow_files = text.strip().split("@@@")
                    print(flow_files, image_path)
                else:
                    text, image_path = text.split("@@")
                image_path = os.path.join(args.img_dir, image_path)
                assert os.path.exists(image_path), image_path
                image = Image.open(image_path).convert("RGB")
                image = image.resize(tuple(reversed(image_size)))
                image = transform(image).unsqueeze(0).to("cuda")
                image = image * 2.0 - 1.0
                image = image.unsqueeze(2).to(torch.bfloat16)
                image = model.encode_first_stage(image, None)
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                pad_shape = (image.shape[0], T - 1, C, H // F, W // F)
                image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)
            else:
                image = None
                if args.flow_from_prompt:
                    text, flow_files = text.split("\t")
            total_num_frames = (T - 1) * 4 + 1  # T is the video latent size, 13 * 4 = 52
            if args.no_flow_injection:
                video_flow = None
            elif args.flow_from_prompt:
                assert args.flow_path is not None, "Flow path must be provided if flow_from_prompt is True"
                p = os.path.join(args.flow_path, flow_files)
                print(f"Flow path: {p}")
                video_flow = (
                    torch.load(p, map_location="cpu", weights_only=True)[:total_num_frames].unsqueeze_(0).cuda()
                )
            elif args.flow_path:
                print(f"Flow path: {args.flow_path}")
                video_flow = torch.load(args.flow_path, map_location=device, weights_only=True)[
                    :total_num_frames
                ].unsqueeze_(0)
            elif args.point_path:
                if type(args.point_path) == str:
                    args.point_path = json.loads(args.point_path)
                print(f"Point path: {args.point_path}")
                video_flow, points = process_traj(args.point_path, total_num_frames, image_size, device=device)
                video_flow = video_flow.unsqueeze_(0)
            else:
                print("No flow injection")
                video_flow = None

            if video_flow is not None:
                model.to("cpu")  # move model to cpu, run vae on gpu only.
                tmp = rearrange(video_flow[0], "T H W C -> T C H W")
                video_flow = flow_to_image(tmp).unsqueeze_(0).to("cuda")  # [1 T C H W]
                if args.vis_traj_features:
                    os.makedirs("samples/flow", exist_ok=True)
                    vis_tensor(tmp, *tmp.shape[-2:], "samples/flow/flow1_vis.gif")
                    imageio.mimwrite(
                        "samples/flow/flow2_vis.gif",
                        rearrange(video_flow[0], "T C H W -> T H W C").cpu(),
                        fps=8,
                        loop=0,
                    )
                del tmp
                video_flow = (
                    rearrange(video_flow / 255.0 * 2 - 1, "B T C H W -> B C T H W").contiguous().to(torch.bfloat16)
                )
                video_flow = video_flow.repeat(2, 1, 1, 1, 1).contiguous()  # for uncondition
                model.first_stage_model.to(device)
                torch.cuda.empty_cache()
                video_flow = model.encode_first_stage(video_flow, None)
                video_flow = video_flow.permute(0, 2, 1, 3, 4).contiguous()
            torch.cuda.empty_cache()
            model.to(device)

            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            if args.image2video and image is not None:
                c["concat"] = image
                uc["concat"] = image

            for index in range(args.num_samples_per_prompt):
                if cnt > 0:
                    args.seed = np.random.randint(1e6)
                    set_random_seed(args.seed)
                # reload model on GPU
                model.to(device)
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    video_flow=video_flow,
                )
                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                # Unload the model from GPU to save GPU memory
                model.to("cpu")
                torch.cuda.empty_cache()
                first_stage_model = model.first_stage_model
                first_stage_model = first_stage_model.to(device)

                latent = 1.0 / model.scale_factor * samples_z

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )

                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                save_path = args.output_dir
                name = str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:60] + f"_{index}_seed{args.seed}"
                if args.flow_from_prompt:
                    name = Path(flow_files).stem
                if mpu.get_model_parallel_rank() == 0:
                    save_video_as_grid_and_mp4(
                        samples,
                        save_path,
                        name,
                        fps=args.sampling_fps,
                        traj_points=locals().get("points", None),
                        prompt=text,
                    )
            del samples_z, samples_x, samples, video_flow, latent, recon, recons, c, uc, batch, batch_uc
            gc.collect()


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
    args.model_config.en_and_decode_n_samples_a_time = 1

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
