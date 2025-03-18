import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
filename = 'test_data/perpendicular.jpg'
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
# Apply Sobel filters to detect long gradients (edges) along X and Y axes
sobel_x = cv.Sobel(img_bilateral, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(img_bilateral, cv.CV_64F, 0, 1, ksize=5)

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

# Use Hough Line Transform to detect long straight lines
lines = cv.HoughLinesP(edges_dilate, 1, np.pi / 180, threshold=200, minLineLength=50, maxLineGap=10)
line_mask = np.zeros_like(edges_dilate)

# Draw only the detected long straight lines onto a mask
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_mask, (x1, y1), (x2, y2), 255, 2)

# Combine the original edges and the line mask to emphasize straight lines
combined_edges = cv.bitwise_or(edges_dilate, line_mask)

lines_blurred = cv.GaussianBlur(src=line_mask, ksize = (7,7), sigmaX=3, sigmaY=3)
corner_response = cv.cornerHarris(np.float32(lines_blurred),45,5,0.2)
 
#result is dilated for marking the corners, not important
corner_response = cv.morphologyEx(corner_response, cv.MORPH_OPEN, kernel, iterations=3)
 
# Threshold for an optimal value, it may vary depending on the image.
im_copy = img.copy()
im_copy[corner_response>0.01*corner_response.max()]=[255,0,0]

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