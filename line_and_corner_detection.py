import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

# Images that currently work:
filename = 'test_data/lines_in_background.jpg' 
# filename = 'test_data/perpendicular.jpg' 
# filename = 'test_data/small_angle_bottom.jpg' 
# filename = 'test_data/small_angle_left.jpg' 

# Images that currently don't work:
# filename = 'test_data/large_angle_right.jpg' 
# filename = 'test_data/small_angle_top.jpg' # Close to working, maybe just need to tweak learning rate or min lines

img = cv.imread(filename)
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 

GAUSSIAN_KERNEL_SIZE = (15,15)
GAUSSIAN_SIGMA = 3
img_blurred = cv.GaussianBlur(
    src=img_gray,
    ksize=GAUSSIAN_KERNEL_SIZE,
    sigmaX=GAUSSIAN_SIGMA,
    sigmaY=GAUSSIAN_SIGMA
)
img_bilateral = cv.bilateralFilter(img_blurred, 10, 75, 75)
THRESHOLD = 182
_, thresholded1 =  cv.threshold(img_bilateral,THRESHOLD,255,cv.THRESH_TOZERO)
# Apply Sobel filters to detect long gradients (edges) along X and Y axes
sobel_x = cv.Sobel(thresholded1, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(thresholded1, cv.CV_64F, 0, 1, ksize=5)

# Combine the gradients and convert to 8-bit
sobel_combined = cv.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(sobel_combined)
_,thresholded = cv.threshold(sobel_combined,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Apply a closing operation to close gaps in the edges (reinforces long lines)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
closed = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)

# Apply Canny edge detection: Adjust thresholds to ignore weaker text edges
edges = cv.Canny(closed, 10, 75)
edges_dilate = cv.morphologyEx(edges, cv.MORPH_DILATE, kernel)
# Iteratively find hough line transform threshold
lines = None
hough_thresh = 1500
MINIMUM_LINES = 40
LEARNING_RATE = 8
while True:
    # Use Hough Line Transform to detect long straight lines
    lines = cv.HoughLines(edges_dilate, 1, np.pi / 180, threshold=hough_thresh)
    num_lines = 0
    if lines is not None:
        num_lines = len(lines)
        print("found", str(num_lines), "lines using thresh:", str(hough_thresh))
        num_more_lines_req = MINIMUM_LINES - num_lines
        if num_more_lines_req <= 0:
            break
        else:
            delta = LEARNING_RATE * num_more_lines_req
            hough_thresh -= delta
    

line_mask = np.zeros_like(edges_dilate)
mask_h, mask_w = line_mask.shape
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + mask_w*(-b)), int(y0 + mask_h*(a)))
        pt2 = (int(x0 - mask_w*(-b)), int(y0 - mask_h*(a)))
        cv.line(line_mask, pt1, pt2, (255,255,255), 3, cv.LINE_AA)

# Combine the original edges and the line mask to emphasize straight lines
combined_edges = cv.bitwise_or(edges_dilate, line_mask)

lines_blurred = cv.GaussianBlur(src=line_mask, ksize = (7,7), sigmaX=3, sigmaY=3)
corner_response = cv.cornerHarris(np.float32(lines_blurred),45,5,0.2)

#result is dilated for marking the corners, not important
corner_response = cv.morphologyEx(corner_response, cv.MORPH_OPEN, kernel, iterations=4)
 
# Threshold for an optimal value, it may vary depending on the image.
im_copy = img.copy()
im_copy[corner_response>0.01*corner_response.max()]=[255,0,0]

corner_points = np.argwhere(corner_response>0.01*corner_response.max())
if corner_points is not None:
    hull = cv.convexHull(corner_points)
    if hull is not None:
        hull = hull.reshape(-1, 2)
        for point in hull:
            y = point[0]
            x = point[1]
            cv.circle(im_copy, (int(x), int(y)), 40, (0, 255, 0), -1)

def draw_on_axis(ax, image, title):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)

fig, axes = plt.subplots(2, 3)
draw_on_axis(axes[0,0], img, "original")
draw_on_axis(axes[0,1], thresholded, "thresholded sobel")
draw_on_axis(axes[0,2], closed, "closed")
draw_on_axis(axes[1,0], edges_dilate, "dilated edges")
draw_on_axis(axes[1,1], line_mask, "Detected lines")
draw_on_axis(axes[1,2], im_copy, "corners?")

plt.show()