import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, convolve2d

# Load a grayscale image and resize
image = Image.open("catsample.png").convert("L")
image = image.resize((256, 256))
image_array = np.array(image, dtype=np.float32) / 255.0

# Define an edge detection kernel (Sobel-like)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

# Perform correlation (kernel is used as-is)
correlated = correlate2d(image_array, kernel, mode='same', boundary='symm')

# Perform convolution (kernel is flipped before applying)
convolved = convolve2d(image_array, kernel, mode='same', boundary='symm')

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_array, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(correlated, cmap='gray')
plt.title("Correlation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(convolved, cmap='gray')
plt.title("Convolution")
plt.axis("off")

plt.tight_layout()
plt.show()