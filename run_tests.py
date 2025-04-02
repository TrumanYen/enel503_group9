import os
import time

from find_paper_corners import find_paper_corners, draw_on_axis
from matplotlib import pyplot as plt
import cv2
import numpy as np


BG_REMOVAL_THRESH = 180
MIN_HOUGH_LINES = 30
DISPLAY_INTERMEDIATE_RESULTS = False

def perspectiveTransform(image, corners):
    H, W = image.shape[:2]  # Get original height and width
    src = np.array(corners, dtype=np.float32)
    #  should be top left, top right, bottom right, bottom left
    dst = np.array([[0, 0],[W - 1, 0],[W - 1, H - 1],[0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(image, M, (W, H))
    return img

warped_images = []
result_images = []
input_paths = os.listdir("test_data")
for file in input_paths:
    if file.endswith(".jpg"):
        file_path = os.path.join("test_data", file)
        print("Testing with: ", file_path)
        start_time = time.time()
        results, hull, corners = find_paper_corners(
                        file_path=file_path,
                        bg_removal_thresh=BG_REMOVAL_THRESH,
                        minimum_hough_lines=MIN_HOUGH_LINES,
                        display_intermediate_results=DISPLAY_INTERMEDIATE_RESULTS)
        result_images.append(results)
        print("Took", str(time.time() - start_time), "seconds")
        print(corners)
        # warp image if there corners detected
        if corners is not None:
            img = cv2.imread(file_path)
            warped_img = perspectiveTransform(img, corners)
            warped_images.append(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))

num_images = len(result_images)  # or however many you have

fig, axes = plt.subplots(2, num_images, figsize=(10 * num_images, 20))  # ⬅️ Bump this up

for i in range(num_images):
    axes[0][i].imshow(result_images[i])
    axes[0][i].set_title("Detected Corners", fontsize=20)
    axes[0][i].axis("off")

    axes[1][i].imshow(warped_images[i])
    axes[1][i].set_title("Warped Perspective", fontsize=20)
    axes[1][i].axis("off")

plt.tight_layout()
plt.show()