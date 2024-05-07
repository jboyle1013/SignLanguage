import os
import json
from tqdm import tqdm

# Function to get a list of video files in a directory
def get_video_files(directory):
    video_files = []
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            video_files.append(file.replace(".mp4", ""))
    return video_files

# Function to filter JSON instances based on local video files
def filter_json(json_data, video_files):
    njson_data = []
    gloss_data = {}
    for entry in tqdm(json_data):
        gloss = entry['gloss']
        filtered_instances = []
        instances = entry['instances']
        train_split = 0
        val_split = 0
        test_split = 0
        for instance in instances:
            if instance["video_id"] in video_files:
                filtered_instances.append(instance)
                if instance['split'] == "train":
                    train_split = train_split+1
                elif instance['split'] == "val":
                    val_split = val_split+1
                else:
                    test_split = test_split+1

        if len(filtered_instances) > 0 and train_split > 2 and val_split > 1 and test_split > 0:
            gloss_data[gloss] = {"Total Instances" :len(filtered_instances),
                                 "Train Instances" : train_split,
                                 "Val Instances" : val_split,
                                 "Test Instances" : test_split
                                 }
            nentry = entry
            nentry["instances"] = filtered_instances
            njson_data.append(nentry)
    return njson_data, gloss_data

# Directory containing video files
video_directory = "raw_videos_mp4"

# File path for JSON data
json_file_path = "start_kit/WLASL_v0.3.json"

# Read JSON data from file
with open(json_file_path, "r") as f:
    json_data = json.load(f)

# Get list of video files in directory
video_files = get_video_files(video_directory)

# Filter JSON instances based on local video files
filtered_json, gloss_dict = filter_json(json_data, video_files)
sorted_gloss_data = dict(sorted(gloss_dict.items(), key=lambda item: item[1]["Total Instances"], reverse=True))

# Save filtered JSON to a new file
with open("filtered_asl_data.json", "w") as f:
    json.dump(filtered_json, f, indent=4)

with open("filtered_asl_gloss_data.json", "w") as f:
    json.dump(sorted_gloss_data, f, indent=4)

print(f'Number of words in filtered set: {len(filtered_json)}')
print("Filtered JSON saved to filtered_asl_data.json")
