import json
import math
import os
import random

import numpy as np

import cv2

import tensorflow as tf


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def compute_difference(x):
    n = len(x)
    diff = np.zeros(n)  # Create a zero vector to store summarized differences

    for i in range(n):
        for j in range(n):
            if i != j:
                diff[i] += abs(x[i] - x[j])  # Sum of absolute differences

    return diff / n  # Normalize by the number of comparisons to keep scale consistent


def read_pose_file(np_paths):

    try:
        fts=[]
        for np_path in np_paths:
            ft = np.load(np_path)

            xy = ft[:, :2]
            fts.append(xy)
        # angles = torch.atan(ft[:, 110:]) / 90
        # ft = torch.cat([xy, angles], dim=1)
        return fts
    except:
        return None



class Sign_Dataset:
    def __init__(self, index_file_path, split, pose_root, sample_strategy='rnd_start', num_samples=25, num_copies=4,
                 img_transforms=None, video_transforms=None, test_index_file=None):
        assert os.path.exists(index_file_path), "Non-existent indexing file path: {}.".format(index_file_path)
        assert os.path.exists(pose_root), "Path to poses does not exist: {}.".format(pose_root)

        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        if type(split) == 'str':
            split = [split]
        self.body_adj_matrix =np.load('dataset/dataset/adj_matrix_body.npy')
        self.left_adj_matrix = np.load('dataset/dataset/adj_matrix_left.npy')
        self.right_adj_matrix = np.load('dataset/dataset/adj_matrix_right.npy')
        self.test_index_file = test_index_file
        self._make_dataset(index_file_path, split)
        self.data_root = 'dataset/dataset'
        self.index_file_path = index_file_path

        self.framename = 'image_{}_keypoints.json'
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples
        self.split = split
        self.img_transforms = img_transforms
        self.video_transforms = video_transforms
        self.pose_root = f'{self.data_root}/{split[0]}/labels/featurematrix'
        self.num_copies = num_copies


    def load_data(self):
        with open(self.index_file_path, 'r') as file:
            data = json.load(file)
        return [d for d in data if d['split'] in self.split]

    def load_bounding_box(self, video_id, frame_indices):
        """
        Loads a consistent bounding box for all frames in a video.

        Args:
        video_id (str): Identifier for the video.

        Returns:
        tuple: Normalized bounding box coordinates (x1, y1, x2, y2).
        """
        bbox_file_path = f'{self.data_root}/{self.split[0]}/labels/bbox/{video_id}'
        for i in frame_indices:
            bbox_path = f'{bbox_file_path}/image_{i:05}_bbox.txt'
            if os.path.exists(bbox_path):
                with open(bbox_path, 'r') as f:
                    line = f.readline().strip()
                    _, x1, y1, x2, y2 = map(int, line.split())
                    # Normalize based on assumed frame dimensions (e.g., 224x224)
                    return (x1 / 128, y1 / 128, x2 / 128, y2 / 128)
            else:
                print(f"Bounding box file {bbox_file_path} not found.")
                return (0, 0, 1, 1)  # Default bounding box (covers the whole frame)


    def load_video_frames(self, video_id, frame_indices, target_size=(128, 128)):
        """
        Loads and resizes saved image frames from specified directories.

        Args:
        video_id (str): The identifier for the video, used to locate the directory.
        frame_indices (list of int): Frame indices to load.
        target_size (tuple of int): The (width, height) to resize frames to.

        Returns:
        numpy.ndarray: An array of resized video frames.
        """
        frames = []
        video_path = f'{self.data_root}/{self.split[0]}/processed/{video_id}'  # Directory where frames are stored

        for i in frame_indices:
            frame_path = os.path.join(video_path, f'image_{i:05}.png')
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)  # Resize the frame
                frames.append(frame)
            else:
                print(f"Frame {frame_path} not found.")

        return np.array(frames)  # Returning as a numpy array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x1 = self.data[index]
        # frames of dimensions (T, H, W, C)
        video_id = x1['video_id']
        frame_start = x1['frame_start']
        frame_end = x1['frame_end']
        gloss_cat = x1['gloss_cat']
        frame_indices = self.get_frame_indices(frame_start, frame_end, self.sample_strategy, self.num_samples)
        frames = self.load_video_frames(video_id, frame_indices)
        body_features, left_features, right_features = self._load_poses(video_id, frame_start, frame_end, self.sample_strategy, self.num_samples)
        bbox = self.load_bounding_box(video_id, frame_indices)
        if self.video_transforms:
            body_features = self.video_transforms(body_features)
            left_features = self.video_transforms(left_features)
            right_features = self.video_transforms(right_features)

        y = gloss_cat

        return frames, body_features, left_features, right_features, self.body_adj_matrix, self.left_adj_matrix, self.right_adj_matrix, np.asarray(bbox), video_id,  y

    def get_frame_indices(self, frame_start, frame_end, sample_strategy, num_samples):
        """Assumes frame indices are to be sampled sequentially for simplicity."""
        return list(range(frame_start, frame_end + 1))[:num_samples]


    def _make_dataset(self, index_file_path, split):
        # Load the training data
        with open(index_file_path, 'r') as file:
            content = json.load(file)

        # Extract glosses and fit label encoder
        glosses = [entry['gloss'] for entry in content]
        self.label_encoder.fit(glosses)
        labels_encoded = self.label_encoder.transform(glosses)

        if self.test_index_file:
            print(f'Trained on {self.index_file_path}, tested on {self.test_index_file}')
            with open(self.test_index_file, 'r') as file:
                content = json.load(file)

        # Convert labels to one-hot encoding
        num_classes = len(set(labels_encoded))
        labels_one_hot = tf.keras.utils.to_categorical(labels_encoded, num_classes)

        # Prepare dataset entries
        for gloss_entry in content:
            gloss = gloss_entry['gloss']
            gloss_cat = labels_one_hot[self.label_encoder.transform([gloss])[0]]

            for instance in gloss_entry['instances']:
                if instance['split'] not in split:
                    continue

                video_id = instance['video_id']
                frame_start = instance['frame_start']
                frame_end = instance['frame_end']

                # Store in the dataset
                self.data.append({
                    'video_id': video_id,
                    'gloss_cat': gloss_cat,
                    'frame_start': frame_start,
                    'frame_end': frame_end
                })
    def _load_poses(self, video_id, frame_start, frame_end, sample_strategy, num_samples):
        """ Load frames of a video. Start and end indices are provided just to avoid listing and sorting the directory unnecessarily.
         """
        body_poses = []
        left_poses = []
        right_poses = []

        if sample_strategy == 'rnd_start':
            frames_to_sample = rand_start_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'seq':
            frames_to_sample = sequential_sampling(frame_start, frame_end, num_samples)
        elif sample_strategy == 'k_copies':
            frames_to_sample = k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples,
                                                                         self.num_copies)
        else:
            raise NotImplementedError('Unimplemented sample strategy found: {}.'.format(sample_strategy))
        p_split = ['body', 'left', 'right']
        for i in frames_to_sample:
            pose_paths = [f'{self.pose_root}/{video_id}/image_{i:05}_body_featurematrix.npy', f'{self.pose_root}/{video_id}/image_{i:05}_hand_left_featurematrix.npy',
                          f'{self.pose_root}/{video_id}/image_{i:05}_hand_right_featurematrix.npy']
            # pose = cv2.imread(frame_path, cv2.COLOR_BGR2RGB)
            try:
                poses = read_pose_file(pose_paths)
            except FileNotFoundError:
                poses = None
            if poses is not None:
                for i in range(len(poses)):
                    if poses[i] is not None:
                        if self.img_transforms:
                            poses[i] = self.img_transforms(poses[i])
                    if i == 0:
                        body_poses.append(poses[i])

                    if i == 1:
                        left_poses.append(poses[i])
                    if i == 2:
                        right_poses.append(poses[i])
        body_across_time = self._pad_the_poses(body_poses, num_samples, frames_to_sample, (25,2))
        left_across_time = self._pad_the_poses(left_poses, num_samples, frames_to_sample, (21,2))
        right_across_time = self._pad_the_poses(right_poses, num_samples, frames_to_sample, (21,2))

        return body_across_time, left_across_time, right_across_time
    def _pad_the_poses(self, poses, num_samples, frames_to_sample, expected_shape):
        if len(poses) < num_samples:
            # If fewer poses were loaded/appended than expected
            last_valid_pose = poses[-1] if poses else np.zeros((expected_shape))
            poses.extend([last_valid_pose] * (num_samples - len(poses)))

        return np.array(poses)


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
        frames_skip = set()

        num_skips = num_frames - num_samples
        interval = num_frames // num_skips

        for i in range(frame_start, frame_end + 1):
            if i % interval == 0 and len(frames_skip) <= num_skips:
                frames_skip.add(i)

        for i in range(frame_start, frame_end + 1):
            if i not in frames_skip:
                frames_to_sample.append(i)
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample


def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []

    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))

    return frames_to_sample


if __name__ == '__main__':
    root = '/home/realclevername/PycharmProjects/SignLanguage/'

    split_file = os.path.join(root, 'dataset/dataset/asl300.json')
    pose_data_root = os.path.join(root, 'WSASL/data/pose_per_individual_videos')

    num_samples = 64

    train_dataset = Sign_Dataset(index_file_path=split_file, split=['train'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None,
                                 num_samples=num_samples)

    val_dataset = Sign_Dataset(index_file_path=split_file, split=['val'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None,
                                 num_samples=num_samples)

    test_dataset = Sign_Dataset(index_file_path=split_file, split=['test'], pose_root=pose_data_root,
                                 img_transforms=None, video_transforms=None,
                                 num_samples=num_samples)

    test = train_dataset.__getitem__(42)

    x = 5
    # train_dataset = train_dataset.map(preprocess_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # # Assuming 'train_dataset' is a tf.data.Dataset object already loaded with the training data
    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (features, labels))  # This is an example; adjust as per your data structure
    #
    # # Shuffle the dataset
    # train_dataset = train_dataset.shuffle(buffer_size=10000)  # buffer_size depends on the data size, adjust accordingly
    #
    # # Batch the dataset
    # train_dataset = train_dataset.batch(64)
    #
    # cnt = 0
    # for batch_idx, data in enumerate(train_data_loader):
    #     print(batch_idx)
    #     x = data[0]
    #     y = data[1]
    #     print(x.size())
    #     print(y.size())
    #
    # print(k_copies_fixed_length_sequential_sampling(0, 2, 20, num_copies=3))
