import os
import cv2
import numpy as np
import nibabel as nib
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

# Dataset configuration
TRAIN_DATASET_PATH = "D:/Hackdata/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"

# Segment classes
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3
}

# Volume and image configuration
VOLUME_SLICES = 100
VOLUME_START_AT = 22  # first slice of volume that we will include
IMG_SIZE = 128


def get_data_paths(dataset_path=TRAIN_DATASET_PATH):
    # Get all directories
    train_and_val_directories = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    # Extract patient IDs from paths
    def pathListIntoIds(dirList):
        x = []
        for i in range(len(dirList)):
            x.append(dirList[i][dirList[i].rfind('/') + 1:])
        return x

    train_and_test_ids = pathListIntoIds(train_and_val_directories)

    # Split data: Train: 68%, Test: 12%, Val: 20%
    train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2, random_state=42)
    train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15, random_state=42)

    print(f"Train length: {len(train_ids)}")
    print(f"Validation length: {len(val_ids)}")
    print(f"Test length: {len(test_ids)}")

    return train_ids, val_ids, test_ids


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, dataset_path=TRAIN_DATASET_PATH,
                 dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        # Initialization
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(self.dataset_path, i)

            # Load FLAIR
            data_path = os.path.join(case_path, f'{i}_flair.nii')
            flair = nib.load(data_path).get_fdata()

            # Load T1CE
            data_path = os.path.join(case_path, f'{i}_t1ce.nii')
            t1ce = nib.load(data_path).get_fdata()

            # Load segmentation mask
            data_path = os.path.join(case_path, f'{i}_seg.nii')
            seg = nib.load(data_path).get_fdata()

            # Process each slice
            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES * c, :, :, 0] = cv2.resize(
                    flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
                )
                X[j + VOLUME_SLICES * c, :, :, 1] = cv2.resize(
                    t1ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
                )
                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START_AT]

        # Convert class 4 to class 3 (enhancing tumor)
        y[y == 4] = 3

        # One-hot encode masks
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Normalize input images
        return X / np.max(X), Y


def create_data_generators(train_ids, val_ids, test_ids, dataset_path=TRAIN_DATASET_PATH):
    training_generator = DataGenerator(train_ids, dataset_path=dataset_path)
    valid_generator = DataGenerator(val_ids, dataset_path=dataset_path)
    test_generator = DataGenerator(test_ids, dataset_path=dataset_path)

    return training_generator, valid_generator, test_generator


def load_single_case(case_path, case_id):
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))

    # Load FLAIR
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case_id}_flair.nii')
    flair = nib.load(vol_path).get_fdata()

    # Load T1CE
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case_id}_t1ce.nii')
    ce = nib.load(vol_path).get_fdata()

    # Process each slice
    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

    # Normalize
    return X / np.max(X)


if __name__ == "__main__":
    # Test the preprocessing module
    print("Testing preprocessing module...")

    # Get data splits
    train_ids, val_ids, test_ids = get_data_paths()

    # Create generators
    train_gen, val_gen, test_gen = create_data_generators(train_ids, val_ids, test_ids)

    print(f"\nData generators created successfully!")
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    print(f"Test batches: {len(test_gen)}")

    # Test loading one batch
    X_batch, Y_batch = train_gen[0]
    print(f"\nBatch shapes:")
    print(f"X: {X_batch.shape}")
    print(f"Y: {Y_batch.shape}")