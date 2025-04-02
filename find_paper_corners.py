import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from sklearn.cluster import KMeans
from scipy.stats import zscore

def draw_on_axis(ax, image, title):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    
def find_best_perpendicular_clusters(dominant_angles):
    """Finds the two clusters that are closest to 90° apart."""
    best_pair_of_labels = None
    best_perpendicularity = float('inf')

    # Check all pairs of the 3 clusters
    for i in range(len(dominant_angles)):
        for j in range(i + 1, len(dominant_angles)):
            angle_diff = abs(dominant_angles[i] - dominant_angles[j])
            angle_diff = min(angle_diff, 180 - angle_diff)  # Account for circular nature of angles
            perpendicularity = abs(angle_diff - 90)  # Closer to 90 is better

            if perpendicularity < best_perpendicularity:
                best_perpendicularity = perpendicularity
                best_pair_of_labels = (i, j)
    print("good angle 0:", str(dominant_angles[best_pair_of_labels[0]]))
    print("good angle 1:", str(dominant_angles[best_pair_of_labels[1]]))
    return best_pair_of_labels

def filter_outlier_lines(lines):
    angles = np.degrees(lines[:, 0, 1])  # Convert theta from radians to degrees
    angles_radians = np.radians(angles)  # Convert degrees to radians if needed
    unit_vectors = np.column_stack((np.cos(angles_radians), np.sin(angles_radians)))

    # Apply K-Means Clustering in 2d space.  2 clusters for expected lines and 1 for any noisy/wrong lines
    NUM_DOMINANT_ANGLE_CANDIDATES = 3
    kmeans = KMeans(n_clusters=NUM_DOMINANT_ANGLE_CANDIDATES, n_init=10)
    cluster_labels = kmeans.fit_predict(unit_vectors)
    
    # Get mean angles for each cluster
    dominant_angles = np.degrees(np.arctan2(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0]))
    dominant_angles = np.mod(dominant_angles, 180)  # Keep within 0-180 degrees
    print("Dominant angles: ", str(dominant_angles))
    best_pair_of_labels = find_best_perpendicular_clusters(dominant_angles)

    good_angle_0 = dominant_angles[best_pair_of_labels[0]]
    good_angle_1 = dominant_angles[best_pair_of_labels[1]]

    line_errors_scored_by_good_angle_0 = np.minimum(
        np.abs((angles - good_angle_0)),
        180 - np.abs((angles - good_angle_0)))
    line_errors_scored_by_good_angle_1 = np.minimum(
        np.abs((angles - good_angle_1)),
        180 - np.abs((angles - good_angle_1)))

    err_thresh_degrees = 20.0
    valid_lines_mask = (line_errors_scored_by_good_angle_0 < err_thresh_degrees) | (line_errors_scored_by_good_angle_1 < err_thresh_degrees)
    filtered_lines = lines[valid_lines_mask]
    return filtered_lines

def find_all_intersections(lines, image_shape):
    """
    Returns:
    - result: Image with intersection points (red dots) and corner points (green dots)
    - hull: corner points (green dots as coordinates)
    """
    # Prep variables as vectors for efficiency
    rhos = lines[:, 0, 0]
    thetas = lines[:, 0, 1]
    cosines = np.cos(thetas)
    sines = np.sin(thetas)
    # Keep denominators and numerators separate to avoid div by zero
    x_intersections_denominators = np.outer(cosines,sines) - np.outer(sines, cosines)
    x_intersections_numerators = np.outer(rhos, sines) - np.outer(sines, rhos)

    intersections = []
    for i_0 in range(0, len(lines)):
        for i_1 in range(0, i_0):
            angle_diff = np.abs(thetas[i_0] - thetas[i_1])
            if angle_diff < 0.52:  # Approx 30 degrees
                continue  # Skip lines that are not at least 60 degrees apart
            if np.abs(x_intersections_denominators[i_0, i_1]) < 1e-10:  # Check for div by zero
                continue 
            sin_theta_0 = sines[i_0]
            if np.abs(sin_theta_0) < 1e-10: # Check for div by zero again
                continue
            x = x_intersections_numerators[i_0, i_1] / x_intersections_denominators[i_0, i_1]
            if (0 < x < image_shape[1]): # Check if x val in frame
                rho_0 = rhos[i_0]
                cos_theta_0 = cosines[i_0]
                y = (rho_0 - (x*cos_theta_0)) / sin_theta_0
                if (0 < y < image_shape[0]): # Check if y value is in frame
                    intersections.append((x,y))

    
    return np.array(intersections)

def find_paper_corners(
        file_path: str, 
        bg_removal_thresh: int = 180,
        minimum_hough_lines: int = 30,
        display_intermediate_results: bool = True,
        ):

    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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
    _, initial_thresholded =  cv.threshold(img_bilateral, bg_removal_thresh, 255,cv.THRESH_TOZERO)
    # Apply Sobel filters to detect long gradients (edges) along X and Y axes
    sobel_x = cv.Sobel(initial_thresholded, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(initial_thresholded, cv.CV_64F, 0, 1, ksize=5)

    # Combine the gradients and convert to 8-bit
    sobel_combined = cv.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(sobel_combined)
    _,sobel_thresholded = cv.threshold(sobel_combined,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Apply a closing operation to close gaps in the edges (reinforces long lines)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(sobel_thresholded, cv.MORPH_CLOSE, kernel)

    # Apply Canny edge detection: Adjust thresholds to ignore weaker text edges
    edges = cv.Canny(closed, 10, 75)
    edges_dilate = cv.morphologyEx(edges, cv.MORPH_DILATE, kernel)
    lines = cv.HoughLines(edges_dilate, 1, np.pi / 180, threshold=800)[:minimum_hough_lines]   
    lines = filter_outlier_lines(lines)

    line_mask = np.zeros_like(edges_dilate)
    intersections = find_all_intersections(lines, line_mask.shape)
    mask_h, mask_w = line_mask.shape
    scale = min(mask_h,mask_w)

    # Draw the lines.  This is purely for debugging
    if lines is not None and display_intermediate_results:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + scale*(-b)), int(y0 + scale*(a)))
            pt2 = (int(x0 - scale*(-b)), int(y0 - scale*(a)))
            cv.line(line_mask, pt1, pt2, (255,255,255), 3, cv.LINE_AA)
        
    
    # Threshold for an optimal value, it may vary depending on the image.
    result = img.copy()
    for intersection in intersections:
        x, y = intersection
        cv.circle(result, (int(x), int(y)), 30, (255, 0, 0), -1)
        cv.circle(line_mask, (int(x), int(y)), 30, (255, 0, 0), -1)

    if intersections is not None:
        hull = cv.convexHull(intersections)
        if hull is not None:
            hull = hull.reshape(-1, 2)
            for point in hull:
                x = point[0]
                y = point[1]
                cv.circle(result, (int(x), int(y)), 40, (0, 255, 0), -1)


    if display_intermediate_results:
        fig, axes = plt.subplots(2, 4, figsize=(15,8))
        draw_on_axis(axes[0,0], img, "original")
        draw_on_axis(axes[0,1], initial_thresholded, "blurred thresholded")
        draw_on_axis(axes[0,2], sobel_combined, "sobel")
        draw_on_axis(axes[0,3], sobel_thresholded, "sobel thresholded")
        draw_on_axis(axes[1,0], closed, "closed")
        draw_on_axis(axes[1,1], edges_dilate, "dilated edges")
        draw_on_axis(axes[1,2], line_mask, "Detected lines")
        draw_on_axis(axes[1,3], result, "corners?")
        plt.tight_layout()
        plt.show()
    return result, hull

if __name__ == "__main__":
    # Images that currently work:
    filename = 'test_data/lines_in_background.jpg' 
    # filename = 'test_data/perpendicular.jpg' 
    # filename = 'test_data/small_angle_bottom.jpg' 
    # filename = 'test_data/small_angle_left.jpg' 

    # Images that currently don't work (need to fix initial thresholding):
    # filename = 'test_data/large_angle_right.jpg' 
    # filename = 'test_data/small_angle_top.jpg'

    find_paper_corners(filename)