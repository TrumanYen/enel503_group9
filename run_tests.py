import os

from find_paper_corners import find_paper_corners


BG_REMOVAL_THRESH = 180
MIN_HOUGH_LINES = 35

for file in os.listdir("test_data"):
    if file.endswith(".jpg"):
        file_path = os.path.join("test_data", file)
        print("Testing with: ", file_path)
        find_paper_corners(
            file_path=file_path,
            bg_removal_thresh=BG_REMOVAL_THRESH,
            minimum_hough_lines=MIN_HOUGH_LINES)
