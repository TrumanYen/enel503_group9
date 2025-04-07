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
    H, W = image.shape[:2]
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0],[W - 1, 0],[W - 1, H - 1],[0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst) # this function requires that the type be float32 so we add dtype to the 
    img = cv2.warpPerspective(image, M, (W, H))
    return img

def adaptiveThreshold(img):
    result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10) # these 2 numbers were tweaked from original 7 and 0 to get more clear text and reduce the random blobs around the text and on the image

    return result

def otsuThreshold(img):
    ret, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return result

class Results:
    def __init__(self,
                 image_with_corners,
                 warped_image,
                 adaptive_result,
                 otsu_result,):
        self.image_with_corners = image_with_corners
        self.warped_image = warped_image
        self.adaptive_result = adaptive_result
        self.otsu_result = otsu_result

results = []
input_paths = os.listdir("test_data")

for file in input_paths:
    if file.endswith(".jpg"):
        file_path = os.path.join("test_data", file)
        #print("Testing with: ", file_path)
        start_time = time.time()
        im_with_corners, hull, corners = find_paper_corners(
            file_path=file_path,
            bg_removal_thresh=BG_REMOVAL_THRESH,
            minimum_hough_lines=MIN_HOUGH_LINES,
            display_intermediate_results=DISPLAY_INTERMEDIATE_RESULTS)
        #print("Took", str(time.time() - start_time), "seconds")
        #print(corners)

        if corners is not None:
            # get warped image
            img = cv2.imread(file_path)
            warped_img = perspectiveTransform(img, corners)
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
            # get thresholded images
            img_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            img_adaptive = adaptiveThreshold(img_gray)
            img_otsu = otsuThreshold(img_gray)
        else: # No result, use blank images
            warped_img = np.zeros_like(im_with_corners)
            img_adaptive = np.zeros_like(im_with_corners)
            img_otsu = np.zeros_like(im_with_corners)

        results.append(Results(image_with_corners=im_with_corners,
                               warped_image=warped_img,
                               adaptive_result=img_adaptive,
                               otsu_result=img_otsu))

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
