import os
import time

from find_paper_corners import find_paper_corners, draw_on_axis
from matplotlib import pyplot as plt


BG_REMOVAL_THRESH = 180
MIN_HOUGH_LINES = 30
DISPLAY_INTERMEDIATE_RESULTS = False

result_images = []
input_paths = os.listdir("test_data")
for file in input_paths:
    if file.endswith(".jpg"):
        file_path = os.path.join("test_data", file)
        print("Testing with: ", file_path)
        start_time = time.time()
        results, hull = find_paper_corners(
                        file_path=file_path,
                        bg_removal_thresh=BG_REMOVAL_THRESH,
                        minimum_hough_lines=MIN_HOUGH_LINES,
                        display_intermediate_results=DISPLAY_INTERMEDIATE_RESULTS)
        result_images.append(results)
        print("Took", str(time.time() - start_time), "seconds")
        
num_rows = 2
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15,8))
display_counter = 0

for i in range(num_rows):
    for j in range(num_cols):
        draw_on_axis(axes[i,j], result_images[display_counter], input_paths[display_counter])
        display_counter += 1

plt.tight_layout()
plt.show()