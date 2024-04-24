import os

import numpy as np
import json


def make_featurematrices(input_file, output_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    body_keypoints = data['people'][0]['pose_keypoints_2d']
    left_hand_keypoints = data['people'][0]['hand_left_keypoints_2d']
    right_hand_keypoints = data['people'][0]['hand_right_keypoints_2d']


if __name__ == "__main__":
    splits = ['train', 'val', 'test']
    for split in splits:
        base_path = f'dataset/dataset/{split}/labels/keypoint'
        save_base_path = f'dataset/dataset/{split}/labels/featurematrix'
        if not os.path.exists(save_base_path):
            os.makedirs(save_base_path)
        for dirpath, dirnames, filenames in os.walk(base_path, topdown=False):
            direct = dirpath.split('/')[-1]
            for file in filenames:
                save_path = f'{save_base_path}/{direct}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                input_fPath = f'{dirpath}/{file}'
                fname = file.split('_')
                file_fame = '_'.join(fname[:1])
                final_fname = f'{file_fame}_featurematrix.npy'
                save_fPath = f'{save_path}/{final_fname}'

