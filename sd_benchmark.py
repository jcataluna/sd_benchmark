# https://pytorch.org/blog/accelerated-diffusers-pt-20/

# noexport
import os
# Prevents torch.compile from mixing up 3090/4090 code
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

from argparse import ArgumentParser, Namespace
from pathlib import Path
import time

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from tqdm import tqdm
import diffusers
import numpy as np
import torch



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--bs", type=int, default=4, help="Batch size (--vae-slicing enables larger batches")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--vae-slicing", action='store_true', default=True, help="VAE slicing allows a bigger batch size")
    parser.add_argument("--no-vae-slicing", action='store_false', dest='vae_slicing')
    parser.add_argument("--safety-checker", action='store_true', default=True, help="Enable NSFW filter")
    parser.add_argument("--no-safety-checker", action='store_false', dest='safety_checker')
    parser.add_argument("--compile", action='store_true', default=True, help="Compile the Unet (faster at the expense of much slower 1st batch)")
    parser.add_argument("--no-compile", action='store_false', dest='compile')
    parser.add_argument("-n", type=int, default=10, help="Test iterations")
    args = parser.parse_args()
    print("Running with args: ", args)

    torch.cuda.set_device(args.gpu)
    torch.set_float32_matmul_precision('high')
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
    pipe.to(torch.device("cuda", index=args.gpu))
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    if args.compile:
        pipe.unet = torch.compile(pipe.unet)
    if args.vae_slicing:
        print("VAE slicing enabled")
        pipe.enable_vae_slicing() # this will prevent decoder from hogging the RAM
    pipe.set_progress_bar_config(disable=True)
    if not args.safety_checker:
        pipe.safety_checker = None # to benchmark just diffusion without nsfw filter

    print(f"Using batch size {args.bs}")

    prompts = ["A horse is riding an astronaut"] * args.bs
    # Force one time compilation (the first batch would take much longer otherwise)
    images = pipe(prompts, guidance_scale=9.0).images
    assert len(images) == args.bs
    dts = []
    for _ in tqdm(range(args.n)):
        t0 = time.monotonic()
        pipe(prompts, guidance_scale=9.0).images
        dts.append((time.monotonic() - t0) / args.bs)

    mean = np.mean(dts)
    stdev = np.std(dts)
    print(f"Mean {mean:.2f} sec/img ± {stdev * 1.96 / np.sqrt(len(dts)):.2f} (95%)")
    
