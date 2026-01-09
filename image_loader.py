import cv2
import sys
from pathlib import Path
#from logger import log_error
import numpy as np


def load_image(image_path: Path, expected_channels=(1, 3)):
    """
    Load image from disk.
    Uses relative paths based on the script location.
    """
    # Get path relative to the script file

    base_path = Path(__file__).parent
    path = base_path / image_path

    try:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load the image
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to load image: {path}")

        # Check shape
        if image.ndim not in (2, 3):
            raise ValueError(f"Invalid image shape {image.shape}: must be 2D (gray) or 3D (color)")

        # Check number of channels
        channels = 1 if image.ndim == 2 else image.shape[2]
        if channels not in expected_channels:
            raise ValueError(f"Unexpected number of channels: {channels}")

        # Check for NaNs
        if np.isnan(image).any():
            raise ValueError(f"Image contains NaN values: {path}")

        return image

    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

        

    return image

def load_images_from_folder(folder_path: Path, expected_channels=(1, 3), extensions=(".png", ".jpg", ".jpeg", ".tif", ".bmp")):
    """
    Load all images from a folder with validation.
    
    Returns:
        dict of {filename: image}
    """
    base_path = Path(__file__).parent
    folder = base_path / folder_path

    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder}")
        sys.exit(1)

    images = {}
    for file in sorted(folder.iterdir()):  # sorted to maintain order
        if file.suffix.lower() in extensions:
            img = load_image(file, expected_channels=expected_channels)
            images[file.name] = img

    if not images:
        print(f"[ERROR] No valid images found in folder: {folder}")
        sys.exit(1)

    return images

