import os
import argparse
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

# Import our custom modules
from preprocessing import (
    get_data_paths,
    create_data_generators,
    TRAIN_DATASET_PATH,
    IMG_SIZE
)
from unet import create_unet_model, CUSTOM_OBJECTS


def setup_callbacks(model_save_path='D:/Hackdata/model/best_model.weights.h5',
                    log_path='D:/Hackdata/logs/training.log',
                    patience=2,
                    min_lr=0.000001):

    callbacks: list[Callback] = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience,
            min_lr=min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        ),
        CSVLogger(log_path, separator=',', append=False)
    ]

    return callbacks


def train_model(epochs=35,
                batch_size=1,
                learning_rate=0.001,
                dataset_path=TRAIN_DATASET_PATH,
                model_save_path='best_model.weights.h5',
                log_path='training.log',
                use_early_stopping=False,
                early_stopping_patience=5):

    print("=" * 80)
    print("Brain Tumor Segmentation - Training")
    print("=" * 80)

    train_ids, val_ids, test_ids = get_data_paths(dataset_path)

    training_generator, valid_generator, test_generator = create_data_generators(
        train_ids, val_ids, test_ids, dataset_path
    )

    print(f"Training batches: {len(training_generator)}")
    print(f"Validation batches: {len(valid_generator)}")
    print(f"Test batches: {len(test_generator)}")

    model = create_unet_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 2),
        learning_rate=learning_rate
    )

    print(f"Total parameters: {model.count_params():,}")

    callbacks = setup_callbacks(
        model_save_path=model_save_path,
        log_path=log_path
    )

    # Add early stopping if requested
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                verbose=1,
                restore_best_weights=True
            )
        )
        print(f"Early stopping enabled with patience={early_stopping_patience}")

    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Model save path: {model_save_path}")
    print(f"Log path: {log_path}")
    print("=" * 80)

    # Clear any previous session
    K.clear_session()

    # Train the model
    history = model.fit(
        training_generator,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_generator
    )

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    print(f"Best model weights saved to: {model_save_path}")
    print(f"Training logs saved to: {log_path}")

    return model, history


def load_best_model(model_path='best_model.weights.h5', learning_rate=0.001):
    print(f"Loading model from {model_path}...")

    # Create model architecture
    model = create_unet_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 2),
        learning_rate=learning_rate
    )

    # Load weights
    model.load_weights(model_path)

    print("Model loaded successfully!")
    return model


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train U-Net for brain tumor segmentation')

    parser.add_argument('--epochs', type=int, default=35,
                        help='Number of training epochs (default: 35)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--dataset-path', type=str, default=TRAIN_DATASET_PATH,
                        help='Path to BraTS dataset')
    parser.add_argument('--save-path', type=str, default='best_model.weights.h5',
                        help='Path to save model weights (default: best_model.weights.h5)')
    parser.add_argument('--log-path', type=str, default='training.log',
                        help='Path to save training logs (default: training.log)')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Patience for early stopping (default: 5)')

    args = parser.parse_args()

    # Train model
    model, history = train_model(
        epochs=args.epochs,
        learning_rate=args.lr,
        dataset_path=args.dataset_path,
        model_save_path=args.save_path,
        log_path=args.log_path,
        use_early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience
    )

    return model, history


if __name__ == "__main__":
    # Run training
    model, history = main()

    print("\n" + "=" * 80)
    print("You can now use the trained model for inference!")
    print("Load the model using: load_best_model('best_model.weights.h5')")
    print("=" * 80)