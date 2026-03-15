"""
NO PREPROCESSING REQUIRED.
"""


def preprocess_all(*args, **kwargs):
    print(
        "\n[Preprocessing] SKIPPED.\n"
        "Your dataset (awsaf49/brats2020-training-data) is pre-processed H5 format.\n"
        "No preprocessing needed — the dataloader reads .h5 files directly.\n"
        "Jump straight to: python main.py --mode train\n"
    )
