# https://pytorch.org/blog/accelerated-diffusers-pt-20/

# noexport
import os
# Prevents torch.compile from mixing up 3090/4090 code
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

from argparse import ArgumentParser, Namespace
from pathlib import Path
import time

from diffusers import FlaxStableDiffusionPipeline
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from tqdm import tqdm
import diffusers
import jax
import jax.numpy as jnp
import numpy as np


def create_key(seed=0):
    return jax.random.PRNGKey(seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--bs", type=int, default=4, help="Batch size (--vae-slicing enables larger batches")
    parser.add_argument("--safety-checker", action='store_true', default=True, help="Enable NSFW filter")
    parser.add_argument("--no-safety-checker", action='store_false', dest='safety_checker')
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("-n", type=int, default=10, help="Test iterations")
    args = parser.parse_args()
    print("Running with args: ", args)

    import jax
    num_devices = jax.device_count()
    device_type = jax.devices()[0].device_kind

    print(f"Found {num_devices} JAX devices of type {device_type}.")
    assert "TPU" in device_type, "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"

    pipe, params = FlaxStableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", 
        dtype=jnp.bfloat16, 
        revision="bf16")
    pipe.set_progress_bar_config(disable=True)
    if not args.safety_checker:
        pipe.safety_checker = None # to benchmark just diffusion without nsfw filter

    print(f"Using batch size {args.bs}")

    rng = create_key(0)
    rng = jax.random.split(rng, jax.device_count())

    prompt = ["A horse is riding an astronaut"] * args.bs * jax.device_count()
    prompt_ids = pipe.prepare_inputs(prompt)

    # replicate params
    p_params = replicate(params)

    # shard prompts
    prompt_ids = shard(prompt_ids)

    # Force one time compilation (the first batch would take much longer otherwise)
    images = pipe(prompt_ids, p_params, rng, jit=True, guidance_scale=9.0, width=args.size, height=args.size).images
    assert len(images) == jax.device_count()
    dts = []
    i = 0
    for _ in tqdm(range(args.n)):
        t0 = time.monotonic()
        images = pipe(prompt_ids, p_params, rng, jit=True, guidance_scale=9.0, width=args.size, height=args.size).images
        dts.append((time.monotonic() - t0) / args.bs)
        images = images.reshape(-1, *images.shape[-3:])
        images = pipe.numpy_to_pil(images)
        for img in images:
            img.save(f'{i:06d}.jpg')
            i += 1

    mean = np.mean(dts)
    stdev = np.std(dts)
    print(f"Mean {mean:.2f} sec/img/TPU ± {stdev * 1.96 / np.sqrt(len(dts)):.2f} (95%)")
    
