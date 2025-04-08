import os
import time
from scan_document import scan_document
from matplotlib import pyplot as plt
import cv2
import numpy as np


results = []
input_paths = os.listdir("test_data")

for file in input_paths:
    if file.endswith(".jpg"):
        file_path = os.path.join("test_data", file)
        results.append(scan_document(file_path, False))

num_images = len(results)
# num_images = 3
fig, axes = plt.subplots(4, num_images, figsize=(10 * num_images, 30))

for i in range(num_images):
    result_i = results[i]
    axes[0][i].imshow(result_i.image_with_corners)
    # axes[0][i].set_title("Detected Corners", fontsize=20)
    axes[0][i].axis("off")

    axes[1][i].imshow(result_i.warped_image)
    # axes[1][i].set_title("Warped Perspective", fontsize=20)
    axes[1][i].axis("off")  

    axes[2][i].imshow(result_i.adaptive_result, cmap='gray')
    # axes[2][i].set_title("Adaptive Thresholding", fontsize=22)
    axes[2][i].axis("off")

    axes[3][i].imshow(result_i.otsu_result, cmap='gray')
    # axes[3][i].set_title("Otsu Thresholding", fontsize=20)
    axes[3][i].axis("off")

plt.tight_layout()
plt.show()
