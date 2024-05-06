import numpy as np
import json
from configs import *

# Load JSON data (replace 'your JSON data here' with actual JSON data)
with open("Data/data-subset/train/labels/keypoints/00415/image_00001_keypoints.json", "r") as f:
    data = json.load(f)
body_keypoints = data['people'][0]['pose_keypoints_2d']
left_hand_keypoints = data['people'][0]['hand_left_keypoints_2d']
right_hand_keypoints = data['people'][0]['hand_right_keypoints_2d']

# Helper function to create adjacency matrix
def create_adjacency_matrix(num_keypoints, edges):
    adjacency_matrix = np.zeros((num_keypoints, num_keypoints), dtype=int)
    for (i, j) in edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # because the graph is undirected
    return adjacency_matrix

# Define body connections (assuming 18 keypoints for the body)
body_edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head to left arm
    (1, 5), (5, 6), (6, 7),          # Right arm
    (1, 8), (8, 9), (9, 10),         # Spine to left leg
    (1, 11), (11, 12), (12, 13),     # Right leg
    (8, 14), (14, 16),               # Left leg further
    (11, 15), (15, 17)               # Right leg further
]

hand_edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Create adjacency matrices
adj_matrix_body = create_adjacency_matrix(18, body_edges)
adj_matrix_left = create_adjacency_matrix(21, hand_edges)  # Assuming 21 keypoints for hands as defined previously
adj_matrix_right = create_adjacency_matrix(21, hand_edges)

print("Adjacency Matrix for Body:")
print(adj_matrix_body)
np.save(f'{DATASET_PATH}/adj_matrix_body.npy', adj_matrix_body)
print("\nAdjacency Matrix for Left Hand:")
print(adj_matrix_left)
np.save(f'{DATASET_PATH}/adj_matrix_left.npy', adj_matrix_left)
print("\nAdjacency Matrix for Right Hand:")
print(adj_matrix_right)
np.save(f'{DATASET_PATH}/adj_matrix_right.npy', adj_matrix_right)

