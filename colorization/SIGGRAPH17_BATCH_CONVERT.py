# simple run command
# python script_name.py -i path/to/input_directory -o path/to/output_directory --use_gpu

import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from colorizers import *

def colorize_directory(input_dir, output_dir, use_gpu=False):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the colorizer
    colorizer = siggraph17(pretrained=True).eval()
    if use_gpu:
        colorizer.cuda()

    # Process each image in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust the extension if needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load and preprocess the image
            img = load_img(input_path)
            tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256,256))
            if use_gpu:
                tens_l_rs = tens_l_rs.cuda()

            # Colorize and save the image
            out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())
            plt.imsave(output_path, out_img_siggraph17)

            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory containing B&W images')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save colorized images')
    parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
    args = parser.parse_args()

    colorize_directory(args.input_dir, args.output_dir, args.use_gpu)
