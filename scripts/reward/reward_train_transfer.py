"""
Train a time-independent reward model via transfer learning from pretrained ResNet18,
based on the human feedback data.
"""

import argparse
import copy
import os
import pickle
import random

import blobfile as bf
import torch as th
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import BinaryImageDatasetFromFeedback
from guided_diffusion.transfer_learning import create_pretrained_model
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    pretrained_ImageNet_defaults
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    log_dir = args.log_dir
    logger.configure(log_dir)
    logger.log(vars(args))

    logger.log("loading pretrained classifier weights...")
    model = create_pretrained_model(
        **args_to_dict(args, pretrained_ImageNet_defaults().keys())
    )
    model.to(dist_util.dev())

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        logger.log(
            f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
        )
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint, map_location=dist_util.dev()
            )
        )
        logger.log("loading successfully completed!")

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=False, initial_lg_loss_scale=16.0
    )
    
    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")

    def load_from_feedback(feedback_dict, batch_size):
        dataset = BinaryImageDatasetFromFeedback(
            feedback_dict=feedback_dict,
            rgb = args.rgb,
        )

        if args.augment_lsun:
            transforms = [
                T.RandomResizedCrop(size=args.image_size, scale=(0.75, 1.0)),
                T.RandomRotation(degrees=30 , interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=(0.75, 1.33), contrast=(0.75, 1.33), saturation=(0.75, 1.33), hue=(-0.2, 0.2)),
            ]
            benign_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomChoice(transforms, p=args.p_benign_transform)
            ])
            malign_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomChoice(transforms, p=args.p_malign_transform)
            ])
            dataset.augment_dataset(
                benign_transform=benign_transform,
                malign_transform=malign_transform,
                augment_data_dir=args.augment_data_dir,
                num_augment=int(args.num_augment),
            )
        
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False
        )

        while True:
            yield from loader

    with open(args.feedback_path, "rb") as f:
        feedback_dict = pickle.load(f)

    data = load_from_feedback(
        feedback_dict, args.batch_size
    )


    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training reward model...")

    loss_fn = th.nn.BCEWithLogitsLoss(pos_weight=args.pos_weight * th.ones([1], device=dist_util.dev()))

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(device=dist_util.dev(), dtype=th.float32)

        batch = batch.to(dist_util.dev())
        
        for i, (sub_batch, sub_labels) in enumerate(
            split_microbatches(args.microbatch, batch, labels)
        ):
            logits = model(sub_batch)
            loss = loss_fn(th.flatten(logits), sub_labels)

            logger.log(f"Loss: {loss.item()}")
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)

        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        log_dir="",
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        image_size=256,
        rgb=True,
        output_dim=1,
        feedback_path="",
        pos_weight=0.1,
        augment_lsun=False,
        augment_data_dir=None,
        num_augment=None,
    )
    defaults.update(pretrained_ImageNet_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        '--p_malign_transform', type=float, nargs=3, default=None,
        help="Sequence of probabilities to apply each transform, for malign samples. Only used for imagenet."
    )
    parser.add_argument(
        '--p_benign_transform', type=float, nargs=3, default=None,
        help="Sequence of probabilities to apply each transform, for benign samples. Only used for imagenet."
    )
    return parser


if __name__ == "__main__":
    main()
