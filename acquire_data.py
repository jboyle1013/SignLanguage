import json
import os
import shutil
import cv2
import yt_dlp  as youtube_dl
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import yaml

# Use this if you are using WSL.
# If you don't know what that is, don't worry about it,
# and remove it from the fstrings, or just swap it out with
# your absolute path or something.
WSL_PATH = "/mnt/c/<rest of file path here>"


# Function to download a video from YouTube using yt_dlp
def download_youtube_video(youtube_url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Specify format for video and audio
        'outtmpl': output_path,  # Template for output file name
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"{youtube_url}"])  # Downloading the video

# Function to save a snippet from a video
def save_video_snippet(source_path, start_time, end_time, snippet_filename):
    try:
        with VideoFileClip(source_path) as video:
            snippet = video.subclip(start_time, end_time)  # Extract the video snippet
            snippet.write_videofile(snippet_filename, codec="libx264")  # Write the snippet to a file
    except Exception as e:
        print(f"Error processing {source_path}: {str(e)}")

# Function to extract and save frames from a video snippet
def extract_and_save_frames(snippet_filename, frames_output_dir, num_frames=30):
    if not os.path.exists(frames_output_dir):
        os.makedirs(frames_output_dir)  # Create output directory if it doesn't exist

    cap = cv2.VideoCapture(snippet_filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, total_frames // num_frames)  # Calculate the number of frames to skip

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))  # Resize frame
            frame_filename = os.path.join(frames_output_dir, f"frame_{i}.png")
            cv2.imwrite(frame_filename, frame)  # Write frame to a file
        else:
            break

    cap.release()  # Release the video capture object

# Function to process JSON data, download videos, extract snippets and frames
def process_data(json_data, snippets_output_dir, frames_output_dir):
    word_usage = {}  # Track how many times each word appears
    for item in tqdm(json_data):  # tqdm shows progress bar
        try:
            # Extracting data from each JSON item
            youtube_url = item['url']
            start_time = item['start_time']
            end_time = item['end_time']
            label = item['label']
            box = item['box']
            word = item['clean_text']

            if label <= 100:  # Using MS-ASL100
                word_usage[word] = word_usage.get(word, 0) + 1

                word_label = f"{word}{word_usage[word]}"

                # Define file paths for various outputs
                downloaded_video_path = f"{WSL_PATH}full_downloads/{word_label}_download.mp4"
                snippet_filename = f"{snippets_output_dir}/videos/{word_label}_snippet.mp4"
                individual_frames_dir = f"{frames_output_dir}/images/{word_label}_frames"

                # Process: download, save snippet, extract frames
                download_youtube_video(youtube_url, downloaded_video_path)
                save_video_snippet(downloaded_video_path, start_time, end_time, snippet_filename)
                extract_and_save_frames(snippet_filename, individual_frames_dir)

                # Writing label data to files
                label_frames_path = f"{frames_output_dir}/labels/{word_label}_frames.txt"
                label_video_path = f"{snippets_output_dir}/labels/{word_label}_snippet.txt"
                label_str = f"{label} {box[0]} {box[1]} {box[2]} {box[3]}"
                with open(label_frames_path, 'w') as file:
                    file.write(label_str)
                with open(label_video_path, 'w') as file:
                    file.write(label_str)

        except Exception as e:
            print(f"Error processing {item}. Error: {e}")

    # Delete Full Downloads Folder after processing
    if os.path.isdir(f"{WSL_PATH}full_downloads"):
        shutil.rmtree(f"{WSL_PATH}full_downloads")

# Create YOLO Style YAML
def create_data_yaml(json_data):
    frames_yaml_data = {
        'train': './data/MS-ASL/frames/train',
        'val': './data/MS-ASL/frames/val',
        'test': './data/MS-ASL/frames/test',
        'names': {}
    }
    video_yaml_data = {
        'train': './data/MS-ASL/video/train',
        'val': './data/MS-ASL/video/val',
        'test': './data/MS-ASL/video/test',
        'names': {}
    }
    names = {}

    for item in tqdm(json_data):
        label = item['label']
        word = item['clean_text']
        if label <= 100:  # Using MS-ASL100
            if word not in names:
                names[label] = word
                names[label] = word

    sorted_names = {key: names[key] for key in sorted(names, key=int)}

    frames_yaml_data['names'] = sorted_names
    video_yaml_data['names'] = sorted_names

    # File path for the YAML file
    file_paths = ['frames_data.yaml', 'video_data.yaml']
    dicts = [frames_yaml_data, video_yaml_data]
    # Writing to the YAML file
    for i in range(len(file_paths)):
        with open(file_paths[i], 'w') as file:
            yaml.dump(dicts[i], file, default_flow_style=False)

        print(f"Data written to {file_paths[i]}")


# Load the JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def acquire_data():
    fLocation = ["test", "train", "val"]
    fps = [f"{WSL_PATH}MS-ASL_jsons/MSASL_test.json", f"{WSL_PATH}MS-ASL_jsons/MSASL_train.json", f"{WSL_PATH}MS-ASL_jsons/MSASL_val.json"]
    for i in range(len(fps)):
        print(f"Processing file: {fps[i]}")
        json_data = load_json(fps[i])
        process_data(json_data, f"{WSL_PATH}data/MS-ASL/video/{fLocation[i]}", f"{WSL_PATH}data/MS-ASL/frames/{fLocation[i]}")
    json_data = load_json(fps[1])
    create_data_yaml(json_data)

if __name__ == '__main__':
    acquire_data()

