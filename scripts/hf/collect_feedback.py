import os
import argparse
import blobfile as bf
from functools import partial
import pickle
from PIL import Image
from PIL import ImageTk

import tkinter as tk
import tkinter.font as tkfont
import tkinter.messagebox as msgbox

class ImageButton(tk.Button):
    def __init__(self, master, img_path, resolution, *args, **kwargs):
        self.img_path = img_path
        self.resolution = resolution
        
        unclicked = Image.open(img_path).resize((resolution, resolution))
        clicked = self.get_clicked_version(unclicked)
        double_clicked = self.get_double_clicked_version(unclicked)
        self.unclickedImage = ImageTk.PhotoImage(unclicked)
        self.clickedImage = ImageTk.PhotoImage(clicked)
        self.doubleclickedImage = ImageTk.PhotoImage(double_clicked)
        
        super().__init__(master, *args, image=self.unclickedImage, **kwargs)
        self.toggleState = 1
        self.bind("<Button-1>", self.clickFunction)
    
    def change_image(self, new_path):
        self.img_path = new_path
        if new_path is None:
            self.unclickedImage = None
            self.clickedImage = None
            self.doubleclickedImage = None
            self.config(image="")
            self["state"] = tk.DISABLED
        else:
            unclicked = Image.open(new_path).resize((self.resolution, self.resolution))
            clicked = self.get_clicked_version(unclicked)
            double_clicked = self.get_double_clicked_version(unclicked)
            self.unclickedImage = ImageTk.PhotoImage(unclicked)
            self.clickedImage = ImageTk.PhotoImage(clicked)
            self.doubleclickedImage = ImageTk.PhotoImage(double_clicked)
            self.toggleState = 1
            self.config(image=self.unclickedImage)
    
    def get_clicked_version(self, unclicked):
        clicked = unclicked.copy()
        linewidth = self.resolution // 16
        for x in range(self.resolution):
            for y in range(self.resolution):
                if linewidth < x < self.resolution - linewidth and linewidth < y < self.resolution - linewidth:
                    continue
                else:
                    clicked.putpixel((x, y), (255, 0, 0))
        return clicked
    
    def get_double_clicked_version(self, unclicked):
        double_clicked = unclicked.copy()
        linewidth = self.resolution // 16
        for x in range(self.resolution):
            for y in range(self.resolution):
                if linewidth < x < self.resolution - linewidth and linewidth < y < self.resolution - linewidth:
                    continue
                else:
                    double_clicked.putpixel((x, y), (0, 0, 255))
        return double_clicked
    
    def clickFunction(self, event=None):
        if self.cget("state") != "disabled":
            self.toggleState = (self.toggleState - 1) % 3
            if self.toggleState == 0:
                self.config(image=self.clickedImage)
            elif self.toggleState == 1:
                self.config(image=self.unclickedImage)
            else:
                self.config(image=self.doubleclickedImage)


def main():

    global feedbacks, curr_filename, image_buttons

    args = create_argparser().parse_args()

    if not os.path.isdir(args.data_dir):
        print("ERROR: Provided data directory does not exist!")
        return 0
    else:
        img_list = _list_image_files_recursively(args.data_dir)
        img_iterator = iter(img_list)

    if os.path.isfile(args.feedback_path):
        with open(args.feedback_path, "rb") as f:
            feedbacks = pickle.load(f)
    else:
        feedbacks = {}

    root = tk.Tk()
    root.title("Feedback data collector")
    root.attributes("-fullscreen", True)

    frame = tk.Frame(root)
    frame.pack()

    question = tk.Label(frame, text=f"Please select all images that contain {args.censoring_feature}.\
    \nRed boundary indicates that the image contains {args.censoring_feature} (for sure).\
    \nBlue boundary indicates that it is undecidable whether the image contains {args.censoring_feature}.", font=tkfont.Font(family="Arial", size=15), pady=10)
    question.pack()

    curr_filename = next(img_iterator)
    while curr_filename in feedbacks.keys():
        curr_filename = next(img_iterator, None)
    
    if curr_filename is None:
        print("Feedback already completed.")
        return 0

    img_grid = tk.Frame(frame)
    img_grid.pack(pady=10)

    image_buttons = []
    idx = 0
    while idx < args.grid_row * args.grid_col and curr_filename is not None:
        image_btn = ImageButton(master=img_grid, img_path=curr_filename, resolution=args.resolution)
        image_buttons.append(image_btn)
        
        curr_filename = next(img_iterator, None)
        idx += 1
    
    for idx, image_btn in enumerate(image_buttons):
        image_btn.grid(row=idx // args.grid_col, column=idx % args.grid_col, padx=3, pady=3)

    def store_response_n_change_images():
        # store feedbacks within the dictionary
        global feedbacks, curr_filename, image_buttons

        for image_btn in image_buttons:
            if image_btn.img_path is None:
                continue
            feedbacks[image_btn.img_path] = None if image_btn.toggleState == 2 else image_btn.toggleState

        if curr_filename is None:
            save_feedbacks(quit_program=True)

        for image_btn in image_buttons:
            image_btn.change_image(curr_filename)
            curr_filename = next(img_iterator, None)

    submit_btn = tk.Button(frame, text="Submit", width=10, height=3, command=store_response_n_change_images)
    submit_btn.pack(pady=5)
    
    def save_feedbacks(quit_program=False):
        # if len(feedbacks.keys()) == len(img_list):
        save = msgbox.askyesnocancel("Save", "Save feedback results?")
        if save:
            with open(args.feedback_path, "wb") as f:
                pickle.dump(feedbacks, f)
            
            malign_count, benign_count = 0, 0
            for path in feedbacks.keys():
                if feedbacks[path] == 0:
                    malign_count += 1
                else:
                    benign_count += 1
            msgbox.showinfo(
                "Save complete", f"Results successfully saved.\
                Collected {len(feedbacks)} feedbacks with {malign_count} malign samples and {benign_count} benign samples!"
            )
        
        if quit_program:
            root.quit()
    
    save_btn = tk.Button(frame, text="Save", width=10, height=3, command=save_feedbacks)
    save_btn.pack(pady=5)

    root.bind("<Escape>", lambda event: save_feedbacks(quit_program=True))
    root.protocol("WM_DELETE_WINDOW", partial(save_feedbacks, True))
    root.config()
    root.mainloop()


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def create_argparser():
    defaults = dict(
        data_dir="",
        feedback_path="",
        censoring_feature="",
        resolution=150,
        grid_row=5,
        grid_col=5,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()