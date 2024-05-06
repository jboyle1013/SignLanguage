from snl_generator import Sign_Dataset
import os
from snl_model import *
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import matplotlib.pyplot as plt

#
# experiment = Experiment(
#   api_key="noN1pXsdgwkmeG9OIwoffbFLO",
#   project_name="snl",
#   workspace="jboyle1013"
# )


def dataset_generator(dataset):
    """Yield processed items from the dataset."""
    for frames, features_body, features_left_hand, features_right_hand, adj_body, adj_left_hand, adj_right_hand, bboxes, video_id, labels in dataset:
        # # Assuming you need to apply preprocessing to frames:
        # processed_frames = np.array([preprocess_frame(frame, bbox, (224, 224)) for frame, bbox in zip(frames, bboxes)])
        yield [frames, features_body, features_left_hand, features_right_hand, adj_body, adj_left_hand, adj_right_hand], labels

def get_datasets():
    root = '/home/realclevername/PycharmProjects/SignLanguage/'

    split_file = os.path.join(root, 'dataset/dataset/asl300.json')
    pose_data_root = os.path.join(root, 'WSASL/data/pose_per_individual_videos')

    num_samples = 64
    num_hand_nodes = 21
    num_body_nodes = 25
    features_per_node = 2

    train_data = Sign_Dataset(index_file_path=split_file, split=['train'], pose_root=pose_data_root,
                              img_transforms=None, video_transforms=None,
                              num_samples=num_samples)

    val_data = Sign_Dataset(index_file_path=split_file, split=['val'], pose_root=pose_data_root,
                            img_transforms=None, video_transforms=None,
                            num_samples=num_samples)

    test_data = Sign_Dataset(index_file_path=split_file, split=['test'], pose_root=pose_data_root,
                             img_transforms=None, video_transforms=None,
                             num_samples=num_samples)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(train_data),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # CNN input
                tf.TensorSpec(shape=(None, num_body_nodes, features_per_node), dtype=tf.float32),  # Body features
                tf.TensorSpec(shape=(None, num_hand_nodes, features_per_node), dtype=tf.float32),  # Left hand features
                tf.TensorSpec(shape=(None, num_hand_nodes, features_per_node), dtype=tf.float32),  # Right hand features
                tf.TensorSpec(shape=(None, num_body_nodes, num_body_nodes), dtype=tf.float32),  # Body adjacency matrix
                tf.TensorSpec(shape=(None, num_hand_nodes, num_hand_nodes), dtype=tf.float32),
                # Left hand adjacency matrix
                tf.TensorSpec(shape=(None, num_hand_nodes, num_hand_nodes), dtype=tf.float32)
                # Right hand adjacency matrix
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Labels
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(test_data),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # CNN input
                tf.TensorSpec(shape=(None, num_body_nodes, features_per_node), dtype=tf.float32),  # Body features
                tf.TensorSpec(shape=(None, num_hand_nodes, features_per_node), dtype=tf.float32),  # Left hand features
                tf.TensorSpec(shape=(None, num_hand_nodes, features_per_node), dtype=tf.float32),  # Right hand features
                tf.TensorSpec(shape=(None, num_body_nodes, num_body_nodes), dtype=tf.float32),  # Body adjacency matrix
                tf.TensorSpec(shape=(None, num_hand_nodes, num_hand_nodes), dtype=tf.float32),
                # Left hand adjacency matrix
                tf.TensorSpec(shape=(None, num_hand_nodes, num_hand_nodes), dtype=tf.float32)
                # Right hand adjacency matrix
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Labels
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(val_data),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # CNN input
                tf.TensorSpec(shape=(None, num_body_nodes, features_per_node), dtype=tf.float32),  # Body features
                tf.TensorSpec(shape=(None, num_hand_nodes, features_per_node), dtype=tf.float32),  # Left hand features
                tf.TensorSpec(shape=(None, num_hand_nodes, features_per_node), dtype=tf.float32),  # Right hand features
                tf.TensorSpec(shape=(None, num_body_nodes, num_body_nodes), dtype=tf.float32),  # Body adjacency matrix
                tf.TensorSpec(shape=(None, num_hand_nodes, num_hand_nodes), dtype=tf.float32),
                # Left hand adjacency matrix
                tf.TensorSpec(shape=(None, num_hand_nodes, num_hand_nodes), dtype=tf.float32)
                # Right hand adjacency matrix
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)  # Labels
        )
    )
    return train_dataset, test_dataset, val_dataset
def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset
    )
    return history

def validate(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()

def main():
    batch_size = 32

    train_dataset, test_dataset, val_dataset = get_datasets()
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    model = SignLanguageModel(cnn_input_shape=(128, 128, 3), hand_num_nodes=21, body_num_nodes=25, features_per_node=2,
                              timesteps=64)

    # hyper_params = {
    #    "learning_rate": 0.001,
    #    "steps": 64,
    #    "batch_size": 32,
    # }
    # experiment.log_parameters(hyper_params)

    train_history = train_model(model, train_dataset, val_dataset, epochs=10, batch_size=32)

    # log_model(experiment, model=model, model_name="TheModel")


if __name__ == '__main__':
    main()


