import argparse
import os
import pickle

import numpy as np

def main(args):
    with open(args.all_feedback_path, "rb") as f:
        all_fb = pickle.load(f)
    
    if os.path.isfile(args.out_feedback_path):
        with open(args.out_feedback_path, "rb") as f:
            out_fb = pickle.load(f)
    else:
        out_fb = {}
    
    malign_imgs = []
    benign_imgs = []
    for img in all_fb.keys():
        if all_fb[img] == 0:
            malign_imgs.append(img)
        elif all_fb[img] == 1:
            benign_imgs.append(img)
    
    for idx in np.random.permutation(len(malign_imgs))[:args.num_malign_samples]:
        out_fb[malign_imgs[idx]] = all_fb[malign_imgs[idx]]

    for idx in np.random.permutation(len(benign_imgs))[:args.num_benign_samples]:
        out_fb[benign_imgs[idx]] = all_fb[benign_imgs[idx]]

    with open(args.out_feedback_path, "wb") as f:
        pickle.dump(out_fb, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_feedback_path", type=str, help="")
    parser.add_argument("--out_feedback_path", type=str, help="")
    parser.add_argument("--num_malign_samples", type=int, help="")
    parser.add_argument("--num_benign_samples", type=int, help="")
    
    args = parser.parse_args()
    main(args)