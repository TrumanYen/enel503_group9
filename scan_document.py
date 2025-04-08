import os
import time
from find_paper_corners import find_paper_corners
from matplotlib import pyplot as plt
import cv2
import numpy as np

BG_REMOVAL_THRESH = 175
MIN_HOUGH_LINES = 30

def approximate_aspect_ratio(corners):
    #  should be top left, top right, bottom right, bottom left
    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]

    top_dist = np.linalg.norm(top_right - top_left)
    left_dist = np.linalg.norm(bottom_left - top_left)
    bottom_dist = np.linalg.norm(bottom_right - bottom_left)
    right_dist = np.linalg.norm(bottom_right - top_right)

    width_avg = (top_dist + bottom_dist) * 0.5
    height_avg = (left_dist + right_dist) * 0.5

    return height_avg / width_avg

def perspectiveTransform(image, corners):
    aspect_ratio = approximate_aspect_ratio(corners)
    original_h, original_w = image.shape[:2]
    W = original_w
    H = int(W * aspect_ratio)
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

def scan_document(file_path, display_intermediate_results):
    im_with_corners, hull, corners = find_paper_corners(
            file_path=file_path,
            bg_removal_thresh=BG_REMOVAL_THRESH,
            minimum_hough_lines=MIN_HOUGH_LINES,
            display_intermediate_results=display_intermediate_results)
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

    return Results(image_with_corners=im_with_corners,
                            warped_image=warped_img,
                            adaptive_result=img_adaptive,
                            otsu_result=img_otsu)