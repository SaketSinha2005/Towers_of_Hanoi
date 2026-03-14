import nibabel as nib
import numpy as np

def load_mri(file_path):

    img = nib.load(file_path)

    data = img.get_fdata()

    # normalize intensities
    data = (data - np.mean(data)) / np.std(data)

    print("MRI shape:", data.shape)

    return data
