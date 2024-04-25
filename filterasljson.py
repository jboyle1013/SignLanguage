import os
import json


def get_subdirectories(directory):
    """Get all subdirectory names within the specified directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def update_json(json_file, subdirectories, output_file):
    """Update JSON file by removing instances where the video_id is not in subdirectories."""
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Iterate over each entry in the list and filter instances
    for entry in data:
        entry['instances'] = [instance for instance in entry['instances'] if instance['video_id'] in subdirectories]

    # Optionally, remove any entries where instances become empty
    updated_data = [entry for entry in data if entry['instances']]

    # Save the updated data back to a JSON file
    with open(output_file, 'w') as file:
        json.dump(updated_data, file, indent=4)

    return updated_data


# Example usage:
splits = ['train', 'test', 'val']
directory = 'dataset/dataset'
json_file = 'WSASL/data/splits/asl300.json'
output_file = 'dataset/dataset/asl300.json'
subd_list = []
for split in splits:
    d = f'{directory}/{split}/frames'
    subd_list.extend(get_subdirectories(d))


updated_data = update_json(json_file, subd_list, output_file)
print("Updated JSON data:", updated_data)
with open(output_file, 'w') as file:
    json.dump(updated_data, file, indent=4)
