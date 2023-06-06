import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv2D(image,
           kernel,
           p=0, s=1):
    rows, cols = image.shape
    krow, kcol = kernel.shape
    row_output = (rows - krow + 2 * p) // s + 1
    col_output = (cols - kcol + 2 * p) // s + 1
    output = np.zeros((row_output, col_output))
    temp = np.zeros((rows + 2 * p, cols + 2 * p))
    if p == 0:
        temp = image.copy()
    else:
        temp[p:-p, p:-p] = image.copy()
    for row in range(0, row_output, s):
        for col in range(0, col_output, s):
            slicing = temp[row: row + krow, col: col + kcol].copy()
            value = np.sum(slicing * kernel, axis=(0, 1))
            output[row // s, col // s] = value
    return output

def moore_neighbor_tracing(binary_image):
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
    directions = [7, 0, 1, 6, -1, 2, 5, 4, 3]
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]

    contour.append((x, y))
    current_x, current_y = x, y
    current_dir = 0

    while True:
        found = False
        for i in range(8):
            new_x = current_x + dx[(current_dir + directions[i]) % 8]
            new_y = current_y + dy[(current_dir + directions[i]) % 8]
            if new_x >= 0 and new_x < binary_image.shape[1] and new_y >= 0 and new_y < binary_image.shape[0] and binary_image[new_y, new_x] == 255 and visited[new_y, new_x] == 0:
                contour.append((new_x, new_y))
                visited[new_y, new_x] = 255
                current_x, current_y = new_x, new_y
                current_dir = (current_dir + directions[i]) % 8
                found = True
                break

        if not found:
            break

    return contour

def nguongtudong(img):
    img_new = img.copy()
    mxn = img_new.shape[0] * img_new.shape[1]
    g, hG = np.unique(img, return_counts=True)
    tG = [hG[0]]
    for i in range(1, len(hG)):
        temp = hG[i] + tG[i-1]
        tG.append(temp)
    ghG = []
    for i in range(len(hG)):
        temp = g[i] * hG[i]
        ghG.append(temp)
    ihi = [ghG[0]]
    for i in range(1, len(hG)):
        temp = ihi[i-1] + ghG[i]
        ihi.append(temp)
    mG = []
    for i in range(len(hG)):
        temp = ihi[i] / tG[i]
        mG.append(temp)
    fG = []
    for i in range(len(hG)):
        temp = (tG[i] * ((mG[i] - max(mG)) ** 2)) / (mxn - tG[i])
        fG.append(temp)
    nguong = g[fG.index(max(fG))]
    img_new[img_new >= nguong] = 255
    img_new[img_new < nguong] = 0
    return img_new

img = cv2.imread("XLAandTGMT\\Buoi8\\Picture3.png", cv2.IMREAD_GRAYSCALE)
threshold = nguongtudong(img)
# _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
output_img = img.copy()

gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
edge_img_gx = conv2D(threshold, gx, p=1)
edge_img_gy = conv2D(threshold, gy, p=1)
edge_img = np.sqrt(np.square(edge_img_gx) + np.square(edge_img_gy))
edge_img = edge_img / np.max(edge_img) * 255
edge_img[edge_img > 0] = 255

contours = moore_neighbor_tracing(edge_img)
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