import os
from PIL import Image
import mediapipe as mp
import numpy as np
import cv2


'''
    This file will be used to resize all pictures to a specified amount and save them to a different directory.
'''

def resize_images(input_dir, output_dir, size=(128, 128)):
    """
    Resize images found in input_dir, saving the resized versions to output_dir.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Walk through the input directory
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            # Check for image files
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Path to the source file
                source_path = os.path.join(subdir, file)
                # Create a similar directory structure in the output directory
                rel_path = os.path.relpath(subdir, input_dir)  # relative path to the subdir
                dest_subdir = os.path.join(output_dir, rel_path)
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir)
                # Path to the output file
                dest_path = os.path.join(dest_subdir, file)

                # Open, resize, and save the image
                img = Image.open(source_path)
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(dest_path)

# Data Paths
input_path = '../Data/data-subset/train/frames'
output_path = '../Data/data-subset/train/processed'

# Call the function
resize_images(input_path, output_path)