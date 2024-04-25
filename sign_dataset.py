import json
import os
from random import random

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


def compute_difference_tf(x):
    # Convert list to tensor for operation in TensorFlow
    x_tensor = tf.constant(x, dtype=tf.float32)
    diffs = []
    for i in range(len(x)):
        diffs.append(x_tensor[i] - x_tensor)
    return tf.stack(diffs)


def read_pose_file_tf(filepath):
    body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}

    # Load JSON data
    try:
        with open(filepath, 'r') as f:
            content = json.load(f)["people"][0]
    except (IndexError, FileNotFoundError):
        return None

    # Process keypoints
    keypoints = content['pose_keypoints_2d'] + content['hand_left_keypoints_2d'] + content['hand_right_keypoints_2d']
    keypoints = [v for i, v in enumerate(keypoints) if i // 3 not in body_pose_exclude]

    x = tf.constant([keypoints[i] for i in range(0, len(keypoints), 3)], dtype=tf.float32)
    y = tf.constant([keypoints[i] for i in range(1, len(keypoints), 3)], dtype=tf.float32)

    # Normalize and scale coordinates
    x = 2 * (x / 256.0 - 0.5)
    y = 2 * (y / 256.0 - 0.5)

    # Compute differences
    x_diff = compute_difference_tf(x) / 2
    y_diff = compute_difference_tf(y) / 2

    # Calculate orientations avoiding division by zero
    orient = tf.math.divide_no_nan(y_diff, x_diff)

    # Stack features
    features = tf.stack([x, y, x_diff, y_diff, orient], axis=1)

    # Save processed data
    save_path = os.path.join('processed_features', os.path.basename(filepath))
    tf.io.write_file(save_path, tf.io.serialize_tensor(features))

    return features


class SignDataset(tf.data.Dataset):
    def __new__(cls, index_file_path, split, pose_root, sample_strategy='rnd_start', num_samples=50, num_copies=4,
                img_transforms=None, video_transforms=None, test_index_file=None):
        assert os.path.exists(index_file_path), "Non-existent indexing file path: {}.".format(index_file_path)
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        # Load and prepare data
        instances = cls.load_and_prepare_data(index_file_path, test_index_file, split)

        # Prepare dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: iter(instances),
            output_signature=(
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # Adjust shape according to actual data
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )

        if video_transforms:
            dataset = dataset.map(video_transforms)

        return dataset

    @staticmethod
    def load_and_prepare_data(index_file_path, test_index_file, split):
        with open(index_file_path, 'r') as f:
            content = json.load(f)

        # Initialize label encoder
        label_encoder = LabelEncoder()
        glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])
        labels = label_encoder.fit_transform(glosses)
        onehot_encoder = OneHotEncoder(categories='auto')
        onehot_labels = onehot_encoder.fit_transform(labels.reshape(-1, 1))

        instances = []
        for gloss_entry in content:
            gloss, gloss_instances = gloss_entry['gloss'], gloss_entry['instances']
            gloss_cat = label_encoder.transform([gloss])[0]

            for instance in gloss_instances:
                if instance['split'] not in split:
                    continue

                video_id = instance['video_id']
                frame_start = instance['frame_start']
                frame_end = instance['frame_end']
                instance_data = (video_id, gloss_cat, frame_start, frame_end)
                instances.append(instance_data)

        return instances

    @staticmethod
    def read_pose_file_tf(pose_path):
        # Assume this function is implemented to read and process pose files
        pass

    @staticmethod
    def video_transforms(data):
        # Example transformation, adjust as needed
        data, label, video_id = data
        return data * 2, label, video_id  # Example: multiply data by 2


def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def sequential_sampling(frame_start, frame_end, num_samples):
    """Keep sequentially ${num_samples} frames from the whole video sequence by uniformly skipping frames."""
    num_frames = frame_end - frame_start + 1
    frames_to_sample = []

    if num_frames > num_samples:
        interval = num_frames / num_samples
        frames_to_sample = [frame_start + int(i * interval) for i in range(num_samples)]
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    """Distribute num_samples over num_copies, spaced by frames."""
    frames_to_sample = []

    if num_samples * num_copies <= frame_end - frame_start + 1:
        step_size = (frame_end - frame_start + 1 - num_samples * num_copies) // (
                    num_copies - 1) if num_copies > 1 else 0
        for i in range(num_copies):
            start = frame_start + i * (num_samples + step_size)
            frames_to_sample.extend(range(start, start + num_samples))
    else:
        repeat_frame = (num_samples * num_copies) // (frame_end - frame_start + 1)
        remainder = (num_samples * num_copies) % (frame_end - frame_start + 1)
        frames_to_sample = list(range(frame_start, frame_end + 1)) * repeat_frame + list(
            range(frame_start, frame_start + remainder))

    return frames_to_sample
