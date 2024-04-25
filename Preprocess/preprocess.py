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


def get_labels(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary to hold video numbers and their labels
    labels_dict = {}

    # Loop through all directories in the input directory
    for video_num in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_num)
        video_path = "labels/bbox".join(video_path.rsplit("frames", 1))
        print(video_path)
        if os.path.isdir(video_path):
            # Get all text files in this directory
            txt_files = [f for f in os.listdir(video_path) if f.endswith('.txt')]
            if txt_files:
                # Path to the first text file
                first_file_path = os.path.join(video_path, txt_files[0])
                # Read the first number from the first file
                with open(first_file_path, 'r') as file:
                    first_line = file.readline()
                    label = int(first_line.split()[0])  # Assuming label is the first number in the line
                    labels_dict[video_num] = label

    # Convert labels dictionary to an array and save to numpy file
    labels_array = np.array([labels_dict[key] for key in sorted(labels_dict.keys())])
    np.save(os.path.join(output_dir, 'labels.npy'), labels_array)

    return labels_dict

# Training Paths
train_input_path = './Data/data-subset/train/frames'
train_output_path = './Data/data-subset/train/processed'

train_labels = './Data/data-subset/train/bbox'
train_labels_out = './Data/data-subset/train'


# Val Paths
val_input_path = './Data/data-subset/val/frames'
val_output_path = './Data/data-subset/val/processed'

val_labels = './Data/data-subset/val/bbox'
val_labels_out = './Data/data-subset/val'


# Testing Paths
test_input_path = './Data/data-subset/train/frames'
test_output_path = './Data/data-subset/train/processed'

test_labels = './Data/data-subset/test/bbox'
test_labels_out = './Data/data-subset/test'



# Call the function
# resize_images(train_input_path, train_output_path)
# resize_images(val_input_path, val_output_path)
# resize_images(test_input_path, test_output_path)

get_labels(train_input_path, train_labels_out)
get_labels(val_input_path, val_labels_out)
get_labels(test_input_path, test_labels_out)