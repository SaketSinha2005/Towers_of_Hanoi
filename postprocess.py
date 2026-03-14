import numpy as np


def extract_features(mask):

    tumor_pixels = np.sum(mask > 0)

    features = {
        "tumor_pixels": int(tumor_pixels)
    }

    return features
