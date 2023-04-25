import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('XLAandTGMT\\Buoi3\\Picture5.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img[img>100] = 255
img[img<=100] = 0

fig, ax = plt.subplots(2, 1, figsize=(5, 5))

kernel_1 = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]])

kernel_2 = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])

# phep co
def erosion(img, kernel):
    img_new = img.copy()
    n = img_new.shape[0]
    m = img_new.shape[1]
    for i in range(1, n-1):
        for j in range(1, m-1):
            if img[i, j] >= 100:
                center_pixel = [i, j]
                center_kernel = [1, 1]
                xRows = [0, 0, 1, -1, 1, -1, 1, -1]
                yCols = [-1, 1, 0, 0, -1, -1, 1, 1]
                count = 0
                for k in range(8):
                    item = [xRows[k], yCols[k]]
                    pixel_img_x = center_pixel[0] + item[0]
                    pixel_img_y = center_pixel[1] + item[1]
                    pixel_kernel_x = center_kernel[0] + item[0]
                    pixel_kernel_y = center_kernel[1] + item[1]
                    if img[pixel_img_x][pixel_img_y] >= 100 and kernel[pixel_kernel_x][pixel_kernel_y] == 1:
                        count += 1
                if count != np.count_nonzero(kernel) - 1:
                    img_new[i, j] = 0
    return img_new

# phep gian
def dilation(img, kernel):
    img_new = img.copy()
    n = img_new.shape[0]
    m = img_new.shape[1]
    for i in range(1, n-1):
        for j in range(1, m-1):
            if img[i, j] >= 100:
                center_pixel = [i, j]
                center_kernel = [1, 1]
                xRows = [0, 0, 1, -1, 1, -1, 1, -1]
                yCols = [-1, 1, 0, 0, -1, -1, 1, 1]
                for k in range(8):
                    item = [xRows[k], yCols[k]]
                    pixel_img_x = center_pixel[0] + item[0]
                    pixel_img_y = center_pixel[1] + item[1]
                    pixel_kernel_x = center_kernel[0] + item[0]
                    pixel_kernel_y = center_kernel[1] + item[1]
                    if kernel[pixel_kernel_x][pixel_kernel_y] == 1:
                        img_new[pixel_img_x][pixel_img_y] = 255
    return img_new

def tinhchinh_img2(img, kernel):
    img_new = erosion(img, kernel)
    img_new = erosion(img_new, kernel)
    img_new = erosion(img_new, kernel)
    img_new = erosion(img_new, kernel)
    img_new = dilation(img_new, kernel)
    img_new = dilation(img_new, kernel)
    img_new = dilation(img_new, kernel)
    img_new = dilation(img_new, kernel)
    img_new = dilation(img_new, kernel)
    img_new = dilation(img_new, kernel)
    return img_new

def tinhchinh_img5(img, kernel):
    img_new = dilation(img, kernel)
    img_new = erosion(img_new, kernel)
    img_new = dilation(img_new, kernel)
    return img_new

def close(img, kernel):
    img_new = dilation(img, kernel)
    img_new = erosion(img_new, kernel)
    return img_new

img_new = tinhchinh_img5(img, kernel_1)

ax[0].imshow(img, cmap='gray')
ax[1].imshow(img_new, cmap='gray')
plt.show()

