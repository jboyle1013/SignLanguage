import os
from PIL import Image
import numpy as np


'''
    This file will be used to get all the labels and save the as npy arrays with the names as the sequence file ID.
    Edited by: Axel, Ben B
'''


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

if '__main__' == __name__:
    # main data dir
    data_dir = "../data/"

    # Training Paths
    train_input_path = data_dir + 'train/frames'
    train_labels_out = data_dir + 'train'


    # Val Paths
    val_input_path = data_dir + 'val/frames'
    val_labels_out = data_dir + 'val'


    # Testing Paths
    test_input_path = data_dir + 'test/frames'
    test_labels_out = data_dir + 'test'


    get_labels(train_input_path, train_labels_out)
    get_labels(val_input_path, val_labels_out)
    get_labels(test_input_path, test_labels_out)