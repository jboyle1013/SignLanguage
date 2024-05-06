# preprocessing script for WLASL dataset
# 1. Convert .swf, .mkv file to mp4.
# 2. Extract YouTube frames and create video instances.

import os
import json
import cv2
import csv

import shutil

import numpy as np
from tqdm import tqdm

BASE_PATH = 'data'
FILENAME = 'wc.csv'
POSE_PATH = 'pose_per_individual_videos/pose_per_individual_videos'


def convert_everything_to_mp4():
    """
    Converts all files to mp4 format.

    Executes the bash script 'swf2mp4.sh' to convert all files to mp4 format using FFmpeg.

    Parameters:
        None

    Returns:
        None

    """
    cmd = 'start_kit/scripts/swf2mp4.sh'

    os.system(cmd)

def createDirectories(paths):
    """
    Creates directories if not exist
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def writeToCSV(filename, header, data):
    """
    Writes the data to the CSV file
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Writing the header
        writer.writerow(header)
        # Writing the data
        for item in data:
            writer.writerow(item)

def video_to_frames(video_path, size=None):
    """

    video_to_frames(video_path, size=None)

    This method takes a video file path as input and generates a list of frames from the video.

    Parameters:
    - video_path (str): The path of the video file.
    - size (tuple): Optional parameter to resize the frames. It should be a tuple of (width, height).

    Returns:
    - frames (list): A list containing frames extracted from the video.

    Example usage:
    ```python
    video_path = "./path/to/video.mp4"
    frames = video_to_frames(video_path, size=(640, 480))
    ```

    Note: This method requires the OpenCV library to be installed in the Python environment.

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
    """
    Converts an array of frames to a video file.

    Parameters:
    - frame_array (list): List of frames to be converted.
    - path_out (str): File path to store the converted video file.
    - size (tuple): Tuple representing the width and height of the video frames.
    - fps (int, optional): Frame rate of the output video. Default is 25.

    Returns:
    - None
    """
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    """
    Extracts a range of frames from a video and returns them as a list.
    If the video has less frames than the expected range, pad it with last frame.
    Parameters:
        src_video_path (str): The path to the source video file.
        start_frame (int): The index of the start frame to extract.
        end_frame (int): The index of the end frame to extract.
    Returns:
        List[Any]: A list of frames extracted from the video, starting from start_frame and ending at end_frame.
    Example:
        >>> extract_frame_as_video('video.mp4', 100, 200)
        [frame_100, frame_101, frame_102, ..., frame_200]
    """

    frames = video_to_frames(src_video_path, (512,512))
    len_frames = len(frames)

    # Pad frames with the last frame if there are not enough frames
    if len_frames < end_frame + 1:
        # Create default frame (copy of the last frame)
        default_frame = np.copy(frames[-1])
        # Calculate padding size
        padding_size = end_frame + 1 - len_frames
        # Add padding frames
        frames.extend([default_frame] * padding_size)

    return frames[start_frame: end_frame + 1]


def write_gloss_occurrences_to_csv(gloss_dict, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Gloss', 'Occurences'])
        for gloss, occur in gloss_dict.items():
            writer.writerow([gloss, occur])


def write_instances_to_json(gloss_dict, instance_json, filename='data/label_key.json'):
    with open(filename, "w") as f:
        json.dump(instance_json, f, indent=4)


def count_files_scandir(directory):
    with os.scandir(directory) as entries:
        return sum(1 for entry in entries if entry.is_file())


def get_middle_indices(total_items):
    if total_items <= 64:
        return 1, 64  # If there are not more than 64 items, return None

    # Calculate the starting index for the middle 64 items
    start_index = (total_items // 2) - 32
    end_index = start_index + 64  # End index is exclusive

    return start_index, end_index - 1  # Adjust to make the end index inclusive

def get_middle_64_indices(start_number, end_number):
    total_items = end_number - start_number + 1

    if total_items <= 64 or end_number < 1:
        end_number = start_number + 63
        return start_number, end_number  # There's no middle segment of 64 if total items are 64 or less

    # Calculate the middle start index
    middle_start_index = (total_items // 2) - 32 + start_number
    middle_end_index = middle_start_index + 63  # Ensure the range includes 64 numbers

    return middle_start_index, middle_end_index

def extract_all_yt_instances(content, numbers, base_path='data'):
    cnt = 1
    filename = 'wc.csv'
    if not os.path.exists('videos'):
        os.mkdir('videos')
    inst_json = []
    sdict = {}
    for _, number in enumerate(tqdm(numbers)):
        entry = content[number]
        gloss = entry['gloss']
        instances = entry['instances']
        inst_size = len(instances)
        sdict[gloss] = inst_size
        index = _
        id_list = []
        with open('data/keys.txt', "r") as f:
            keys = f.read().splitlines()

        if gloss not in keys:
            with open('data/keys.txt', "a") as f:
                f.write(f"{index+1} : {gloss}\n")


        for inst in instances:
            url = inst['url']
            video_id = inst['video_id']

            yt_identifier = url[-11:]
            if 'youtube' in url or 'youtu.be' in url:
                src_video_path = os.path.join(f'raw_videos_mp4', yt_identifier + '.mp4')

            else:
                src_video_path = os.path.join(f'raw_videos_mp4', video_id + '.mp4')

            dst_video_path = os.path.join(f'{base_path}/videos', video_id + '.mp4')

            if not os.path.exists(src_video_path):
                src_video_path = os.path.join(f'raw_videos_mp4', video_id + '.mp4')

            if os.path.exists(src_video_path):

                id_list.append(video_id)


                cnt += 1

                split = inst['split']
                bbox = inst['bbox']
                frame_base_path = f'{base_path}/{split}/frames/{video_id}'
                labels_key_base_path = f'{base_path}/{split}/labels/keypoint/{video_id}'
                labels_box_base_path = f'{base_path}/{split}/labels/bbox/{video_id}'
                if not os.path.exists(frame_base_path):
                    os.makedirs(frame_base_path)
                if not os.path.exists(labels_key_base_path):
                    os.makedirs(labels_key_base_path)
                if not os.path.exists(labels_box_base_path):
                    os.makedirs(labels_box_base_path)


                # because the JSON file indexes from 1.
                start_frame = inst['frame_start'] - 1
                end_frame = inst['frame_end'] - 1
                nstart, nend = get_middle_64_indices(start_frame, end_frame)
                last_keypoints = None
                kc = count_files_scandir(f'pose_per_individual_videos/{video_id}')
                kns, kne = get_middle_indices(kc)

                selected_frames = extract_frame_as_video(src_video_path, nstart, nend)
                for _, frame in enumerate(selected_frames):
                    n = _ + 1
                    kn = _ + kns
                    cv2.imwrite(f'{base_path}/{split}/frames/{video_id}/image_{n:05}.png', frame)

                    keypoint_file_path = f'pose_per_individual_videos/{video_id}/image_{kn:05}_keypoints.json'
                    labels_box_path = f'{base_path}/{split}/labels/bbox/{video_id}/image_{n:05}_bbox.txt'

                    # Set last or default keypoints
                    if not os.path.isfile(keypoint_file_path):
                        keypoints = last_keypoints
                    else:
                        with open(keypoint_file_path, 'r') as file:
                            keypoints = json.load(file)
                            last_keypoints = keypoints

                    # Write keypoints to new location
                    keypoint_copy_path = f'{base_path}/{split}/labels/keypoint/{video_id}/image_{n:05}_keypoints.json'
                    with open(keypoint_copy_path, 'w') as file:
                        json.dump(keypoints, file)

                    # Set last or default bbox
                    if _ > len(selected_frames) - 1:
                        bbox = last_bbox

                    label_str = f"{index} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
                    last_bbox = bbox

                    # Write bbox to new location
                    with open(labels_box_path, 'w') as file:
                        file.write(label_str)

        gloss_dict = {'number': index + 1, 'videos': id_list}
        inst_json.append({gloss: gloss_dict})
    with open('data/label_key.json', "w") as f:
        json.dump(inst_json, f, indent=4)
        # # when OpenCV reads an image, it returns size in (h, w, c)
        #     # when OpenCV creates a writer, it requres size in (w, h).
        #     size = selected_frames[0].shape[:2][::-1]
        #
        #     convert_frames_to_video(selected_frames, dst_video_path, size)

            # print(cnt, dst_video_path)


def delete_empty_subfolders(path='data'):
    """
    Deletes empty subfolders in the specified directory.

    :param path: The path to the directory containing the subfolders. Default is 'data'.
    :type path: str
    :return: None
    """
    if not os.path.isdir(path):
        print(f"Error: The specified path '{path}' is not a directory.")
        return
    split = ['train', 'val', 'test']
    base_path = 'data'
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
    """
    1. Convert .swf, .mkv file to mp4.
    2. Load JSON content from 'WLASL_v0.3.json' file.
    3. Extract all YouTube instances from the JSON content.

    This method is the entry point of the program and performs the following tasks:
    - Converts specific video files to MP4 format
    - Loads a JSON file
    - Extracts YouTube instances from the JSON content

    Returns:
        None
    """
    # 1. Convert .swf, .mkv file to mp4.
    # convert_everything_to_mp4()
    numbers = [33, 35, 45, 48, 50, 53, 67, 68, 76, 80, 139, 146, 152, 175, 197, 257, 266, 270, 271, 278]

    content = json.load(open('start_kit/WLASL_v0.3.json'))
    extract_all_yt_instances(content, numbers)
    delete_empty_subfolders()

if __name__ == "__main__":
    main()
