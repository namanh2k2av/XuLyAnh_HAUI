import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loc trung binh

img = cv2.imread('XLAandTGMT\\Buoi2\\Picture2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(2, 1, figsize=(5, 5))

H = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]) / 9.0

def conv(img, kernel):
    img_new = np.copy(img)
    n = img_new.shape[0]
    m = img_new.shape[1]
    for i in range(1, n-1):
        for j in range(1, m-1):
            center_pixel = [i, j]
            center_kernel = [1, 1]
            xRows = [0, 0, 1, -1, 1, -1, 1, -1]
            yCols = [-1, 1, 0, 0, -1, -1, 1, 1]
            new_val = 0.0
            for k in range(8):
                item = [xRows[k], yCols[k]]
                pixel_img_x = center_pixel[0] + item[0]
                pixel_img_y = center_pixel[1] + item[1]
                pixel_kernel_x = center_kernel[0] + item[0]
                pixel_kernel_y = center_kernel[1] + item[1]
                new_val += img[pixel_img_x][pixel_img_y] * kernel[pixel_kernel_x][pixel_kernel_y]
            new_val += img[i][j] * kernel[1][1]
            if new_val < 0:
                new_val = 0
            if new_val > 255:
                new_val = 255
            img_new[i, j] = new_val * 1
    return img_new
img_new = conv(img,H)
for i in range(0,2):
    img_new = conv(img_new,H)

ax[0].imshow(img, cmap='gray')
ax[1].imshow(img_new, cmap='gray')


plt.show()