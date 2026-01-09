import numpy as np

def add_gaussian_noise(img, mean=0.0, sigma=20.0, iterations = 1):
    """
    Add additive gaussian noise to an image
    
    Parameters:
        img: np.ndarray
            Input image (grayscale or BGR).
        mean: int
        sigma: float
    
    Returns:
        noisy: np.ndarray
            Image with noise
    """
    for _ in range(iterations):
        noise = np.random.normal(mean, sigma, img.shape)

        noisy = img.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255)

    return noisy.astype(img.dtype)

def simulate_illumination(img, brightness_factor=1.0, bias=0, noise_std=0):
    """
    Simulate illumination variations on an image.
    
    Parameters:
        img: np.ndarray
            Input image (grayscale or BGR).
        brightness_factor: float
            Multiplicative factor to simulate overall brightening/darkening.
            e.g., 1.2 brightens, 0.8 darkens.
        bias: int
            Additive intensity shift. Positive to brighten, negative to darken.
        noise_std: float
            Standard deviation of Gaussian noise to simulate uneven lighting.
    
    Returns:
        img_illum: np.ndarray
            Image with simulated illumination variations (same dtype as input).
    """
    # Convert to float for calculations
    img_float = img.astype(np.float32)
    
    # Apply multiplicative factor
    img_float *= brightness_factor
    
    # Apply additive bias
    img_float += bias
    
    # Optionally add Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
        img_float += noise
    
    # Clip values to valid range [0,255]
    img_float = np.clip(img_float, 0, 255)
    
    # Convert back to original dtype
    return img_float.astype(img.dtype)