# preprocessing script for WLASL dataset
# 1. Convert .swf, .mkv file to mp4.
# 2. Extract YouTube frames and create video instances.

import os
import json
import cv2

import shutil

from tqdm import tqdm


def convert_everything_to_mp4():
    cmd = 'bash scripts/swf2mp4.sh'

    os.system(cmd)


def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []
    
    while True:
        ret, frame = cap.read()
    
        if ret:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            break

    cap.release()

    return frames


def convert_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    frames = video_to_frames(src_video_path)

    return frames[start_frame: end_frame+1]


def extract_all_yt_instances(content, base_path='WLASL-master/WLASL/data'):
    cnt = 1
    pose_path = 'WLASL-master/pose_per_individual_videos/pose_per_individual_videos'
    if not os.path.exists('videos'):
        os.mkdir('videos')

    for _, entry in enumerate(tqdm(content)):
        gloss = entry['gloss']
        instances = entry['instances']
        index = _

        with open('WLASL-master/WLASL/data/keys.txt', "r") as f:
            keys = f.read().splitlines()

        if gloss not in keys:
            with open('WLASL-master/WLASL/data/keys.txt', "a") as f:
                f.write(f"{index+1} : {gloss}\n")


        # for inst in instances:
        #     url = inst['url']
        #     video_id = inst['video_id']
        #     split = inst['split']
        #     bbox = inst['bbox']
        #     frame_base_path = f'{base_path}/{split}/frames/{video_id}'
        #     labels_key_base_path = f'{base_path}/{split}/labels/keypoint/{video_id}'
        #     labels_box_base_path = f'{base_path}/{split}/labels/bbox/{video_id}'
        #     if not os.path.exists(frame_base_path):
        #         os.makedirs(frame_base_path)
        #     if not os.path.exists(labels_key_base_path):
        #         os.makedirs(labels_key_base_path)
        #     if not os.path.exists(labels_box_base_path):
        #         os.makedirs(labels_box_base_path)
        #
        #     cnt += 1
        #
        #     yt_identifier = url[-11:]
        #     if 'youtube' in url or 'youtu.be' in url:
        #         src_video_path = os.path.join(f'{base_path}/raw_videos_mp4', yt_identifier + '.mp4')
        #
        #     else:
        #         src_video_path = os.path.join(f'{base_path}/raw_videos', video_id + '.mp4')
        #
        #
        #     dst_video_path = os.path.join(f'{base_path}/videos', video_id + '.mp4')
        #
        #
        #     if not os.path.exists(src_video_path):
        #         src_video_path = os.path.join(f'{base_path}/raw_videos_mp4', video_id + '.mp4')
        #
        #     if os.path.exists(dst_video_path):
        #         print('{} exists.'.format(dst_video_path))
        #         continue
        #
        #     # because the JSON file indexes from 1.
        #     start_frame = inst['frame_start'] - 1
        #     end_frame = inst['frame_end'] - 1
        #
        #     if end_frame <= 0:
        #         shutil.copyfile(src_video_path, dst_video_path)
        #         continue
        #
        #     selected_frames = extract_frame_as_video(src_video_path, start_frame, end_frame)
        #     for _, frame in enumerate(selected_frames):
        #         number = _+start_frame+1
        #         cv2.imwrite(f'{base_path}/{split}/frames/{video_id}/image_{number:05}.png', frame)
        #         shutil.copy(f'WLASL-master/pose_per_individual_videos/pose_per_individual_videos/{video_id}/image_{number:05}_keypoints.json',
        #                     f'{base_path}/{split}/labels/keypoint/{video_id}/image_{number:05}_keypoints.json')
        #
        #         label_str = f"{index} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
        #
        #         labels_box_path = f'{base_path}/{split}/labels/bbox/{video_id}/image_{number:05}_bbox.txt'
        #
        #         with open(labels_box_path, 'w') as file:
        #             file.write(label_str)

            # when OpenCV reads an image, it returns size in (h, w, c)
            # when OpenCV creates a writer, it requres size in (w, h).
            # size = selected_frames[0].shape[:2][::-1]
            #
            # convert_frames_to_video(selected_frames, dst_video_path, size)

            # print(cnt, dst_video_path)

def delete_empty_subfolders(path='WLASL-master/WLASL/data'):
    """Deletes all empty subdirectories within the specified directory."""
    if not os.path.isdir(path):
        print(f"Error: The specified path '{path}' is not a directory.")
        return
    split = ['train', 'val', 'test']
    base_path = 'WLASL-master/WLASL/data'
    for split in split:
        spath = f'{base_path}/{split}'
        fpath = f'{spath}/frames'
        lpath = f'{spath}/labels'
    # Loop through the directory tree from the bottom up
        for dirpath, dirnames, filenames in os.walk(lpath, topdown=False):
            for dirname in dirnames:
                full_dir_path = f'{dirpath}/{dirname}'
                # Check if the directory is empty
                if not os.listdir(full_dir_path):
                    os.rmdir(full_dir_path)
                    print(f"Deleted empty folder: {full_dir_path}")

        
def main():
    # 1. Convert .swf, .mkv file to mp4.
    convert_everything_to_mp4()

    content = json.load(open('WLASL_v0.3.json'))
    extract_all_yt_instances(content)


if __name__ == "__main__":
    main()

