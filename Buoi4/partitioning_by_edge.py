import numpy as np
import cv2

# Custom Sobel operator for edge detection
def sobel_operator(image):
    # Define Sobel kernels for x and y direction
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Pad the image with zeros
    padded_image = np.pad(image, 1, mode='constant')

    # Calculate gradients in x and y directions
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)

    rows, cols = image.shape
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            region = padded_image[i-1:i+2, j-1:j+2]
            gradient_x[i-1, j-1] = np.sum(region * sobel_x)
            gradient_y[i-1, j-1] = np.sum(region * sobel_y)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient_magnitude

# Thresholding to create binary image
def threshold_image(image, threshold):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image >= threshold] = 255
    return binary_image

# Custom image partitioning by edge finding
def partition_image(image, threshold):
    # Perform edge detection
    gradient_magnitude = sobel_operator(image)

    # Thresholding to create binary image
    binary_image = threshold_image(gradient_magnitude, threshold)

    # Invert binary image
    inverted_image = cv2.bitwise_not(binary_image)

    # Perform connected component analysis to partition the image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image)

    # Get the largest connected component (excluding the background)
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    largest_component = (labels == largest_component_label).astype(np.uint8) * 255

    return largest_component

# Read the input image
image_path = r"C:\Users\LENOVO\Documents\Image_Processing\img test th4\anhbai4_3.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Set the threshold value for edge detection
threshold = 50

# Partition the image
partitioned_image = partition_image(image, threshold)

# Display the original image and partitioned image
cv2.imshow("Original Image", image)
cv2.imshow("Partitioned Image", partitioned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
