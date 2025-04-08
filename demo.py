from scan_document import scan_document
from matplotlib import pyplot as plt

# filename = 'test_data/lines_in_background.jpg' 
# filename = 'test_data/perpendicular.jpg' 
# filename = 'test_data/small_angle_bottom.jpg' 
# filename = 'test_data/small_angle_left.jpg' 
filename = 'test_data/large_angle_right.jpg' 
# filename = 'test_data/small_angle_top.jpg'
# filename = 'test_data/rotated_and_perspective.jpg' 
# filename = 'test_data/rotated_and_perpendicular.jpg' 

result = scan_document(filename, True)

num_cols = 4
fig, axes = plt.subplots(1, num_cols, figsize=(10 * num_cols, 30))

axes[0].imshow(result.image_with_corners)
axes[0].set_title("Detected Corners", fontsize=20)
axes[0].axis("off")

axes[1].imshow(result.warped_image)
axes[1].set_title("Warped Perspective", fontsize=20)
axes[1].axis("off")  

axes[2].imshow(result.adaptive_result, cmap='gray')
axes[2].set_title("Adaptive Thresholding", fontsize=22)
axes[2].axis("off")

axes[3].imshow(result.otsu_result, cmap='gray')
axes[3].set_title("Otsu Thresholding", fontsize=20)
axes[3].axis("off")

plt.tight_layout()
plt.show()
