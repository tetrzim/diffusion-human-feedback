"""
Performs censored sampling based on trained reward model(s).
Unifies the previously scattered sampling scripts.

TODO:
Let's first integrate the ensemble vs. non-emsemble
Then time-dependent classifier vs. resnet-based classifier
Then finally the ldm cases
"""

import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "latent_guided_diffusion"))

import argparse
import ctypes
from time import time, localtime, strftime

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
from torchvision.utils import make_grid, save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    pretrained_ImageNet_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.transfer_learning import create_pretrained_model
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad

def setup_time_dependent_reward(args):
    kwargs = args_to_dict(args, classifier_defaults().keys())
    kwargs["output_dim"] = 1
    return create_classifier(**kwargs)

def setup_time_independent_reward(args):
    kwargs = args_to_dict(args, pretrained_ImageNet_defaults().keys())
    kwargs["output_dim"] = 1
    return create_pretrained_model(**kwargs)
    
REWARD_SETUP_IDS = {
    "time_dependent": id(setup_time_dependent_reward),
    "time_independent": id(setup_time_independent_reward),
}


class OptimizerDetails:
    def __init__(self):
        self.num_recurrences = None
        self.operation_func = None
        self.optimizer = None # handle it on string level
        self.lr = None
        self.loss_func = None
        self.backward_steps = 0
        self.loss_cutoff = None
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = None
        self.tv_loss = None
        self.use_forward = False
        self.forward_guidance_wt = 0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = None
        self.loss_save = None


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.log_dir)
    time_tag = strftime("%m%d_%I:%M:%S", localtime(time()))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.use_ldm:
        model = diffusion
        model.load_state_dict(
            th.load(args.model_path)["state_dict"],
            strict=False
        )
    else:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        if args.use_fp16:
            model.convert_to_fp16()
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading reward models...")
    # Load reward models
    # reward_builder is the function that creates a template for the reward model
    # depending on whether it is time-dependent or not (in which case transfer learning is used)
    reward_list = []
    if args.reward_paths is not None:
        for i, reward_path in enumerate(args.reward_paths):
            reward_builder = ctypes.cast(REWARD_SETUP_IDS["time_dependent" if args.time_dependent_reward else "time_independent"], ctypes.py_object).value
            reward = reward_builder(args)
            reward.load_state_dict(
                dist_util.load_state_dict(reward_path, map_location="cpu")
            )
            reward.to(dist_util.dev())
            if args.classifier_use_fp16:
                reward.convert_to_fp16()
            reward.eval()
            reward_list.append(reward)


    """
    Loads classifier if classifier guidance should be used in generation
    """
    if args.target_class is not None and args.classifier_scale > 1e-6:
        logger.log("loading classifier for class-oriented guidance...")
        classifier_args = args_to_dict(args, classifier_defaults().keys())
        classifier_args["output_dim"] = args.classifier_output_dim
        classifier = create_classifier(**classifier_args)
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

    log_sigmoid = th.nn.LogSigmoid()
    
    # For original guidance
    def cond_fn(x, t, y=None):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # Computes the sum of log probabilities from each reward (ensembling)
            # NOTE: This is no longer the average as in the previous version.
            #       Updated in order to maintain consistency with the paper.
            log_probs = sum([log_sigmoid(reward(x_in, t)) for reward in reward_list])
            out = log_probs.sum() * args.original_guidance_wt

            # Adds the classifier guidance term when needed.
            if args.classifier_scale > 1e-6:
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                out += selected.sum() * args.classifier_scale

            return th.autograd.grad(out, x_in)[0]

    def model_fn(x, t, y=None, args=None, model=None):
        return model(x, t, y if args.class_cond else None)

    def operation_func(x, t=None):
        return [reward(x) if t is None else reward(x, t) for reward in reward_list]

    def loss_func(reward_vals, *args):
        return sum([-log_sigmoid(rval) for rval in reward_vals])

    def generate_sample_batch(diffusion):
        
        if args.use_ldm:
            sample_fn = DDIMSamplerWithGrad(diffusion).sample_operation
            with diffusion.ema_scope():
                # LSUN
                samples_ddim, _ = sample_fn(
                    int(args.timestep_respacing), 
                    args.batch_size,
                    ( 
                        diffusion.model.diffusion_model.in_channels,
                        diffusion.model.diffusion_model.image_size,
                        diffusion.model.diffusion_model.image_size,
                    ),
                    operated_image=None,
                    operation=operation,
                    eta=0.,
                    verbose=False,
                )
            sample = diffusion.decode_first_stage(samples_ddim)
        else:
            model_kwargs = {}
            if args.target_class is None:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
            else:
                classes = int(args.target_class) * th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int64)
            model_kwargs["y"] = classes

            sample_fn = diffusion.ddim_sample_loop_operation
            sample = sample_fn(
                partial(model_fn, model=model, args=args),
                (args.batch_size, args.image_channels, args.image_size, args.image_size), # self.shape,
                operated_image=None, 
                operation=operation,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                progress=args.progressive
            )
        
        return sample

    ############### Guidance operation components ###############
    operation = OptimizerDetails()
    operation.num_recurrences = args.num_recurrences
    operation.operation_func = operation_func
    operation.loss_func = loss_func

    # Determines whether we use original guidance
    # (The plain classifier-guidance style guidance)
    operation.original_guidance = args.original_guidance
    operation.sampling_type = args.sampling_type

    # Determines whether we use forward universal guidance
    # (Based on the expectation of the generated x0)
    operation.use_forward = args.use_forward 
    operation.forward_guidance_wt = args.forward_guidance_wt

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr 
    operation.backward_steps = args.backward_steps
    operation.loss_cutoff = args.optim_loss_cutoff
    
    # Other miscellaneous setups
    operation.other_guidance_func = None
    operation.other_criterion = None
    operation.tv_loss = args.optim_tv_loss
    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 10
    operation.folder = logger.get_dir() # results_folder
    if args.optim_print:
        os.makedirs(f'{operation.folder}/samples', exist_ok=True)
    operation.Aug = args.optim_aug
    #############################################################

    logger.log("sampling... ")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        # Run reverse diffusion (or latent diffusion) to generate batch of samples
        sample = generate_sample_batch(diffusion)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    hparam_tag = f"[bwd{args.backward_steps}_lr_{args.optim_lr}]"
    if args.use_forward:
        hparam_tag = hparam_tag + f"_[fwd_wt_{args.forward_guidance_wt}]"
    if args.original_guidance:
        hparam_tag = hparam_tag + f"_[org_wt_{args.original_guidance_wt}]"

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{hparam_tag}_{time_tag}_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
    
    logger.log("sampling complete")

    sample_tensor = th.tensor(np.transpose(arr.astype(np.float32) / 255., (0, 3, 1, 2)))
    sample_save_dir = os.path.join(args.log_dir, f"sample_imgs_{hparam_tag}_{time_tag}")
    if not os.path.isdir(sample_save_dir) and dist.get_rank() == 0:
        os.makedirs(sample_save_dir)

    dist.barrier()
    if dist.get_rank() == 0:
        for i in range(sample_tensor.size(0)):
            save_image(sample_tensor[i], os.path.join(sample_save_dir, f"{i:04d}.png"))


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        model_path="",
        classifier_path="",
        classifier_output_dim=1000,
        classifier_scale=0.0,
        image_channels=3,
        target_class=None,
        time_dependent_reward=True,
        log_dir="",
        # operation
        num_recurrences=1,
        optim_lr=0.01,
        backward_steps=0, # multi-gpu sampling is not supported for positive backward_steps
        optim_loss_cutoff=0.0, 
        optim_tv_loss=False, 
        use_forward=True,
        original_guidance=False,
        original_guidance_wt=0.0,
        forward_guidance_wt=1.0,
        sampling_type='ddpm', # ['ddpm, 'ddim']
        optim_warm_start=False,
        optim_print=False,
        progressive=False,
        optim_aug=None,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(pretrained_ImageNet_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--reward_paths", type=str, nargs='+', 
                        help="list of paths to each reward model"
    )
    return parser


if __name__ == "__main__":
    main()