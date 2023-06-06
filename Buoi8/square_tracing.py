import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv2D(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

def square_tracing(binary_image):
    contours = []
    height, width = binary_image.shape
    visited = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 255 and visited[y, x] == 0:
                contour = trace_contour(binary_image, visited, x, y)
                contours.append(contour)

    return contours

def trace_contour(binary_image, visited, x, y):
    contour = []
    stack = [(x, y, 0)]

    while stack:
        current_x, current_y, side = stack.pop()
        contour.append((current_x, current_y))
        visited[current_y, current_x] = 255

        directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        for i in range(4):
            next_x = current_x + directions[side % 4][0]
            next_y = current_y + directions[side % 4][1]

            if next_x >= 0 and next_x < binary_image.shape[1] and next_y >= 0 and next_y < binary_image.shape[0] and binary_image[next_y, next_x] == 255 and visited[next_y, next_x] == 0:
                stack.append((next_x, next_y, (side + 1) % 4))
                break

            side = (side + 1) % 4

    return contour

def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0 / pixel_number
    his, bins = np.histogram(gray, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        # print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    print(final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img

img = cv2.imread("XLAandTGMT\\Buoi8\\Picture3.png", cv2.IMREAD_GRAYSCALE)
threshold = otsu(img)
# _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
output_img = img.copy()

gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
edge_img_gx = conv2D(threshold, gx)
edge_img_gy = conv2D(threshold, gy)
edge_img = np.sqrt(np.square(edge_img_gx) + np.square(edge_img_gy))
edge_img = edge_img / np.max(edge_img) * 255
edge_img[edge_img > 0] = 255

contours = square_tracing(edge_img)
print(contours)
rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for points in contours:
    for point in points:
        _y, _x = point
        rgb[_x, _y, 0] = 0
        rgb[_x, _y, 1] = 0
        rgb[_x, _y, 2] = 255

cv2.imshow("origin", img)
cv2.imshow("threshold", threshold)
cv2.imshow("edge", edge_img)
cv2.imshow("final", rgb)
cv2.waitKey(0)