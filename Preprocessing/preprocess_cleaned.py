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
    Convert all files to MP4 format.

    This method uses the swf2mp4.sh script to convert all files to MP4 format.

    :return: None
    """
    cmd = 'start_kit/scripts/swf2mp4.sh'

    os.system(cmd)




def writeToCSV(filename, header, data):
    """
    Writes data to a CSV file.

    :param filename: The name of the CSV file to be written.
    :param header: The list of column headers for the CSV file.
    :param data: The list of row data to be written to the CSV file.
    :return: None
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
    Converts a video file to a list of frames.

    :param video_path: Path to the video file.
    :param size: Optional. A tuple (width, height) specifying the desired size of the frames. Default is None.
    :return: A list of frames extracted from the video.

    Example:
    >>> frames = video_to_frames('path/to/video.mp4', (640, 480))
    >>> for frame in frames:
    ...     cv2.imshow('Frame', frame)
    ...     cv2.waitKey(1)

    Note: Make sure to release the video capture object using cap.release() after using this method.
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

    :param frame_array: Array of frames to be converted.
    :param path_out: Path to save the converted video file.
    :param size: Size of the frames in (width, height) format.
    :param fps: Frames per second of the output video. Default is 25.

    :return: None
    """
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    """
    Extracts a range of frames from a video and returns them as a list.

    :param src_video_path: Path to the source video file
    :param start_frame: Start frame index to extract (inclusive)
    :param end_frame: End frame index to extract (inclusive)
    :return: List of extracted frames

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
    """
    Writes the gloss occurrences from the given gloss dictionary to a CSV file.

    :param gloss_dict: The dictionary containing the gloss occurrences.
    :param filename: The name of the CSV file to be written.
    :return: None
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Gloss', 'Occurences'])
        for gloss, occur in gloss_dict.items():
            writer.writerow([gloss, occur])


def write_instances_to_json(gloss_dict, instance_json, filename='data/label_key.json'):
    """

    """
    with open(filename, "w") as f:
        json.dump(instance_json, f, indent=4)


def count_files_scandir(directory):
    """
    Count the number of files in a given directory.

    :param directory: The directory to scan.
    :type directory: str
    :return: The number of files in the given directory.
    :rtype: int
    """
    with os.scandir(directory) as entries:
        return sum(1 for entry in entries if entry.is_file())


def get_middle_indices(total_items):
    """
    :param total_items: The total number of items to be considered.
    :return: The starting and ending indices for the middle 64 items within the total_items range.

    If there are not more than 64 items, None is returned.

    The function calculates the starting index for the middle 64 items based on the total_items parameter.
    The end_index is then calculated by adding 64 to the start_index.
    The returned start_index and end_index are adjusted to make the end index inclusive, by subtracting 1 from the end_index.

    Example usage:

    total_items = 100
    start_index, end_index = get_middle_indices(total_items)
    print(start_index)  # Output: 18
    print(end_index)  # Output: 81
    """
    if total_items <= 64:
        return 1, 64  # If there are not more than 64 items, return None

    # Calculate the starting index for the middle 64 items
    start_index = (total_items // 2) - 32
    end_index = start_index + 64  # End index is exclusive

    return start_index, end_index - 1  # Adjust to make the end index inclusive

def get_middle_64_indices(start_number, end_number):
    """

    """
    total_items = end_number - start_number + 1

    if total_items <= 64 or end_number < 1:
        end_number = start_number + 63
        return start_number, end_number  # There's no middle segment of 64 if total items are 64 or less

    # Calculate the middle start index
    middle_start_index = (total_items // 2) - 32 + start_number
    middle_end_index = middle_start_index + 63  # Ensure the range includes 64 numbers

    return middle_start_index, middle_end_index


def process_all(content, base_path='datasetsolid31', trimming=False):
    """
    Process_all method processes the given content and performs certain operations. It takes the following parameters:

    :param content: A list of entries to process
    :param base_path: The base path to use for file operations. Default is 'datasetsolid31'.
    :param trimming: A boolean value indicating whether trimming should be performed. Default is False.

    :return: None

    """
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(base_path, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    # Initialize containers
    inst_json = []
    sdict = {}

    # Read keys from file
    keys_file_path = os.path.join(base_path, 'keys.txt')
    keys = read_keys(keys_file_path)

    # Loop through content
    for index, entry in enumerate(tqdm(content)):
        process_entry(entry, index, base_path, keys, sdict, inst_json, trimming)

    # Save the instance JSON
    write_instances_to_json(inst_json, os.path.join(base_path, 'label_key.json'))

    # Append new keys to file
    append_new_keys(keys, keys_file_path)


def process_entry(entry, index, base_path, keys, sdict, inst_json, trimming):
    """
    Process a single gloss entry.

    :param entry: The gloss entry to process.
    :param index: The index of the gloss entry.
    :param base_path: The base path for storing the processed information.
    :param keys: The list of keys for storing glosses.
    :param sdict: A dictionary to store the number of instances for each gloss.
    :param inst_json: A list to store the processed information.
    :param trimming: The trimming option for processing instances.
    :return: None
    """
    gloss = entry['gloss']
    instances = entry['instances']

    # Record the number of instances for each gloss
    sdict[gloss] = len(instances)
    id_list = []

    for inst in instances:
        if process_instance(inst, gloss, index, base_path, id_list, trimming):
            continue  # Skip to next instance if processing was successful

    # Save the processed information
    if id_list:
        inst_json.append({gloss: {'number': index + 1, 'videos': id_list}})
        if gloss not in keys:
            keys.append(gloss)  # Prepare to add new gloss to keys.txt


def process_instance(inst, gloss, index, base_path, id_list, trimming):
    """Process an instance.

    :param inst: Dictionary representing the instance.
    :param gloss: Gloss for the instance.
    :param index: Index for the instance.
    :param base_path: Base path for file operations.
    :param id_list: List of ids.
    :param trimming: Trimming for the instance.
    :return: True if processing successful, False otherwise.
    """
    video_id = inst['video_id']
    src_video_path = get_video_path(inst, base_path)
    if not os.path.exists(src_video_path):
        return False

    # Prepare destination paths
    prepare_directories(gloss, video_id, inst['split'], base_path)

    # Process video frames and annotations
    process_video_frames(inst, src_video_path, gloss, video_id, index, base_path, inst['split'], trimming)
    id_list.append(video_id)
    return True


def get_video_path(inst, base_path):
    """
    Constructs the path for the video file based on the provided instance and base path.

    :param inst: Dictionary containing information about the video.
    :type inst: dict
    :param base_path: The base directory path where the video files are stored.
    :type base_path: str
    :return: The constructed video file path.
    :rtype: str
    """
    yt_identifier = inst['url'][-11:]
    video_id = inst['video_id']
    raw_videos_path = os.path.join(base_path, 'raw_videos_mp4')
    if 'youtube' in inst['url'] or 'youtu.be' in inst['url']:
        return os.path.join(raw_videos_path, yt_identifier + '.mp4')
    return os.path.join(raw_videos_path, video_id + '.mp4')


def prepare_directories(gloss, video_id, split, base_path):
    """

    """
    paths = [
        os.path.join(base_path, split, 'frames', f'{gloss}.{video_id}'),
        os.path.join(base_path, split, 'labels', 'keypoint', f'{gloss}.{video_id}'),
        os.path.join(base_path, split, 'labels', 'bbox', f'{gloss}.{video_id}')
    ]
    createDirectories(paths)


def process_video_frames(inst, src_video_path, gloss, video_id, index, base_path, split, trimming):
    """
    :param inst: A dictionary containing information about the frame range to process. It should have keys 'frame_start' and 'frame_end' indicating the starting and ending frame indices
    *.
    :param src_video_path: The path to the source video file.
    :param gloss: The gloss associated with the video frames being processed.
    :param video_id: The ID of the video.
    :param index: The index of the frame being processed.
    :param base_path: The base path where the processed frames will be saved.
    :param split: The split of the dataset (e.g., 'train', 'test', 'val').
    :param trimming: A boolean indicating whether trimming should be applied to the frame range.
    :return: None

    This method processes video frames by extracting the selected frames from the source video, saving them in the specified location, and applying additional operations such as writing
    * keypoints and bounding boxes to new locations.

    The method takes in various parameters such as 'inst' for frame range information, 'src_video_path' for the path to the source video, 'gloss' for the associated gloss, 'video_id' for
    * the video ID, 'index' for the frame index, 'base_path' for the base path where the processed frames will be saved, 'split' indicating the split of the dataset, and 'trimming' to enable
    */disable trimming of frame range.

    Example usage:
        inst = {'frame_start': 10, 'frame_end': 20}
        src_video_path = 'path/to/video.mp4'
        gloss = 'jump'
        video_id = 'video001'
        index = 1
        base_path = 'processed_data'
        split = 'train'
        trimming = True

        process_video_frames(
            inst,
            src_video_path,
            gloss,
            video_id,
            index,
            base_path,
            split,
            trimming
        )
    """
    # Frame extraction and other processing here
    start_frame = inst['frame_start'] - 1
    end_frame = inst['frame_end'] - 1
    last_keypoints = None
    keypoint_count = count_files_scandir(f'pose_per_individual_videos/{video_id}')
    if trimming:
        start_frame, end_frame = get_middle_64_indices(start_frame, end_frame)
        key_start, key_end = get_middle_indices(keypoint_count)

    selected_frames = extract_frame_as_video(src_video_path, start_frame, end_frame)
    for _, frame in enumerate(selected_frames):
        n = _ + 1
        key_num = n
        if trimming:
            key_num = _ + key_start
        cv2.imwrite(f'{base_path}/{split}/frames/{gloss}.{video_id}/image_{n:05}.png', frame)

        keypoint_file_path = f'pose_per_individual_videos/{video_id}/image_{key_num:05}_keypoints.json'
        labels_box_path = f'{base_path}/{split}/labels/bbox/{gloss}.{video_id}/image_{n:05}_bbox.txt'

        # Set last or default keypoints
        if not os.path.isfile(keypoint_file_path):
            keypoints = last_keypoints
        else:
            with open(keypoint_file_path, 'r') as file:
                keypoints = json.load(file)
                last_keypoints = keypoints

        # Write keypoints to new location
        keypoint_copy_path = f'{base_path}/{split}/labels/keypoint/{gloss}.{video_id}/image_{n:05}_keypoints.json'
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



def read_keys(filepath):
    """
    Read the content of a file at the specified filepath and return the lines as a list.

    :param filepath: The absolute or relative path to the file to be read.
    :return: A list containing the lines of the file.
    """
    with open(filepath, "r") as file:
        return file.read().splitlines()


def append_new_keys(keys, filepath):
    """
    Append the given keys to a file.

    :param keys: A list of keys to append.
    :type keys: list
    :param filepath: The path to the file.
    :type filepath: str
    :return: None
    """
    with open(filepath, "a") as file:
        for key in keys:
            file.write(f"{key}\n")


def createDirectories(paths):
    """
    :param paths: A list of paths for which to create directories.
    :return: None

    Create directories for each path specified in the list. If a directory already exists, it is skipped. The `exist_ok=True` argument ensures that no exception is raised if the directory
    * already exists.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def delete_empty_subfolders(path='datasetsolid31'):
    """
    Delete empty subfolders

    :param path: The path to the directory where the empty subfolders will be deleted. Default is 'datasetsolid31'.
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
        for dirpath, dirnames, filenames in os.walk(fpath, topdown=False):
            for dirname in dirnames:
                full_dir_path = f'{dirpath}/{dirname}'
                # Check if the directory is empty
                if not os.listdir(full_dir_path):
                    os.rmdir(full_dir_path)
                    try:
                        os.rmdir(f'{lpath}/{dirname}')
                    except:
                        pass
                    print(f"Deleted empty folder: {full_dir_path}")
        for dirpath, dirnames, filenames in os.walk(lpath, topdown=False):
            for dirname in dirnames:
                full_dir_path = f'{dirpath}/{dirname}'
                # Check if the directory is empty
                if not os.listdir(full_dir_path):
                    os.rmdir(full_dir_path)
                    try:
                        os.rmdir(f'{fpath}/{dirname}')
                    except:
                        pass
                    print(f"Deleted empty folder: {full_dir_path}")


def main():
    """
    Executes the main functionality of the program.

    This method converts .swf and .mkv files to mp4 format using the convert_everything_to_mp4() function.
    It then processes the content of the 'filtered_asl_data.json' file using the process_all() function.
    After that, it removes any empty subfolders using the delete_empty_subfolders() function.

    :return: None
    """
    # 1. Convert .swf, .mkv file to mp4.
    # Will only run in Unix Environment
    convert_everything_to_mp4()


    content = json.load(open('filtered_asl_data.json'))
    process_all(content)
    # delete_empty_subfolders()

if __name__ == "__main__":
    main()
