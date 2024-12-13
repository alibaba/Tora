import argparse
import tempfile
from pathlib import Path
from typing import Literal

import cv2
import imageio
import numpy as np
import torch
from diffusers import CogVideoXDPMScheduler
from diffusers.schedulers import CogVideoXDDIMScheduler
from diffusers.utils import load_image, load_video
from einops import rearrange, repeat
from tora.i2v_pipeline import ToraImageToVideoPipeline
from tora.t2v_pipeline import ToraPipeline
from tora.traj_utils import process_traj
from torchvision.utils import flow_to_image


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
    return video


def export_to_video(video_frames, output_video_path: str = None, fps: int = 10, traj_points=None):
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    video_frames = [np.array(frame) for frame in video_frames]
    video_frames = np.stack(video_frames)
    if traj_points is not None:
        video_frames = draw_points(video_frames, traj_points)

    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path


def generate_video(
    args: argparse.Namespace,
    prompt: str,
    model_path: str,
    flow: torch.Tensor,
    points: None,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 1234,
    fps: int = 8,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    if generate_type == "i2v":
        pipe = ToraImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = ToraPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        raise NotImplementedError("generate_type must be t2v or i2v. v2v not implemented yet.")

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if args.enable_slicing:
        pipe.vae.enable_slicing()
    if args.enable_tiling:
        pipe.vae.enable_tiling()
    if args.enable_compile:
        pipe.transformer = torch.compile(pipe.transformer, backend="inductor")
        pipe.transformer.is_compiled = True

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            video_flow=flow,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            video_flow=flow,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    export_to_video(video_generate, output_path, fps, points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX1.5-5B", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=720, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=480, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=8, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="i2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        default=False,
        help="Whether to enable model-wise CPU offloading when performing validation/testing to save GPU memory.",
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        default=False,
        help="Whether to enable sequential CPU offloading to save lots of GPU memory. (Super slow)",
    )
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether to use VAE slicing for saving GPU memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether to use VAE tiling for saving GPU memory.",
    )
    parser.add_argument(
        "--enable_sageattention",
        action="store_true",
        default=False,
        help="Whether to use SageAttention.",
    )
    parser.add_argument(
        "--enable_compile",
        action="store_true",
        default=False,
        help="Whether to use torch.compile to speed up inference.",
    )

    # for tora flow
    parser.add_argument("--flow_path", type=str, default=None, help="The path of the flow")
    parser.add_argument("--no_flow_injection", action="store_true", default=False, help="No flow injection")
    parser.add_argument(
        "--point_path", type=str, nargs="+", default="cli", help="The path of the provided trajectory points"
    )

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    total_num_frames = args.num_frames
    image_size = (args.width, args.height)

    if args.enable_sageattention:
        import torch.nn.functional as F
        from sageattention import sageattn

        F.scaled_dot_product_attention = sageattn

    if args.no_flow_injection:
        print("No flow injection")
        video_flow = None
        points = None
    elif args.flow_path:
        print(f"Flow path: {args.flow_path}")
        video_flow = torch.load(args.flow_path, map_location="cpu", weights_only=True)[:total_num_frames].unsqueeze_(0)
        points = None
    elif args.point_path:
        print(f"Point path: {args.point_path}")
        video_flow, points = process_traj(args.point_path, total_num_frames, (args.height, args.width), device="cpu")
        video_flow = video_flow.unsqueeze_(0)
    else:
        print("No flow and points provided.")
        video_flow = None
        points = None

    if video_flow is not None:
        tmp = rearrange(video_flow[0], "T H W C -> T C H W")
        video_flow = flow_to_image(tmp).unsqueeze_(0).to("cuda", dtype)  # [1 T C H W]
        del tmp
        video_flow = video_flow / (255.0 / 2) - 1

    generate_video(
        args=args,
        prompt=args.prompt,
        model_path=args.model_path,
        flow=video_flow,
        points=points,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
    )

    print(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3 :.2f} GiB")
