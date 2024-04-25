import os

import numpy as np
import json


def make_featurematrices(input_file, output_file, part):
    with open(input_file, 'r') as file:
        data = json.load(file)
    try:
        if part == 'body':
            keypoints = data['people'][0]['pose_keypoints_2d']
        if part == 'hand_left':
            keypoints = data['people'][0]['hand_left_keypoints_2d']
        if part == 'hand_right':
            keypoints = data['people'][0]['hand_right_keypoints_2d']

        coordinates = np.array(keypoints).reshape(-1, 3)[:, :2]
        np.save(output_file, coordinates)
    except IndexError:
        print(f'No keypoints found in {input_file}')


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
                file_fame = '_'.join(fname[:2])
                file_delname = '_'.join(fname[:1])

                parts = ['body', 'hand_left', 'hand_right']
                for part in parts:
                    if os.path.exists(f'{save_path}/{file_delname}_{part}_featurematrix.npy'):
                        os.remove(f'{save_path}/{file_delname}_{part}_featurematrix.npy')

                    final_fname = f'{file_fame}_{part}_featurematrix.npy'
                    save_fPath = f'{save_path}/{final_fname}'
                    make_featurematrices(input_fPath, save_fPath, part)

