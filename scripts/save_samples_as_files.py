import argparse
import os

from PIL import Image
import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results


def get_samplewise_norm(arr):
    return np.sum(arr ** 2, axis=(1, 2, 3))


def main():
    args = create_argparser().parse_args()

    sample_path = args.sample_path
    sample_arr = np.load(sample_path)["arr_0"] # This is an array of integers w/in range 0~255.
    sample_arr = sample_arr.astype(np.float32) / 255.
    sample_tensor = torch.tensor(np.transpose(sample_arr, (0, 3, 1, 2)))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    for i in range(sample_tensor.size(0)):
        save_image(sample_tensor[i], os.path.join(args.save_dir, f"{i:04d}.png"))
    

def create_argparser():
    defaults = dict(
        sample_path="",
        save_dir="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    
    
    
