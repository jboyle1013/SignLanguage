import os
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

## PREPROCESS ----> bbox, 32x32, grayscale <---- Prone to change
def convert_bbox(y0, x0, y1, x1, width, height):
    """Convert normalized bounding box values to pixel coordinates."""
    x0_pixel = int(x0 * width)
    y0_pixel = int(y0 * height)
    x1_pixel = int(x1 * width)
    y1_pixel = int(y1 * height)
    return x0_pixel, y0_pixel, x1_pixel, y1_pixel

def process_folder(folder_path, label_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) != 30:
        return []  # Skip folders that do not have exactly 30 images

    tensors = []
    with open(label_path, 'r') as file:
        bbox_data = file.read().strip().split()
        _, y0, x0, y1, x1 = map(float, bbox_data)  # Skip the class label

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        x0_pixel, y0_pixel, x1_pixel, y1_pixel = convert_bbox(y0, x0, y1, x1, width, height)
        cropped_image = image[y0_pixel:y1_pixel, x0_pixel:x1_pixel]
        
        # Resize the image to 32x32 pixels using cv2
        resized_image = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_AREA)

        # Convert the OpenCV image to a tensor
        image_tensor = transforms.ToTensor()(resized_image)
        tensors.append(image_tensor)
    
    return tensors

# Define the paths
images_directory = "../data/data/MS-ASL/frames/test/images"
labels_directory = "../data/data/MS-ASL/frames/test/labels"

# Process each folder
for folder_name in os.listdir(images_directory):
    folder_path = os.path.join(images_directory, folder_name)
    label_path = os.path.join(labels_directory, f"{folder_name}.txt")
    tensors = process_folder(folder_path, label_path)
    print(f"Processed tensors from {folder_name}, count: {len(tensors)}")


# --------- Simple CNN ---------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # First convolutional layer taking 1 input channel (image), and producing 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # An adaptive average pool to reduce the spatial dimensions to a size that matches the output of the convolutional layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers / Dense layers
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply conv followed by ReLU, then pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output for the dense layer
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the number of classes based on your dataset
num_classes = 1000
model = SimpleCNN(num_classes)

# TODO write train_model method()
# Loss function