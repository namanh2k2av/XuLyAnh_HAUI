import cv2
import numpy as np
import matplotlib.pyplot as plt
# tim bien
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

prewitt_Hx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])

prewitt_Hy = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])

sobel_Hx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

sobel_Hy = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

H1 = np.array([[5, 5, -3],
               [5, 0, -3],
               [-3, -3, -3]])

H2 = np.array([[5, 5, 5],
               [-3, 0, -3],
               [-3, -3, -3]])

H3 = np.array([[-3, 5, 5],
               [-3, 0, 5],
               [-3, -3, -3]])

H4 = np.array([[-3, -3, 5],
               [-3, 0, 5],
               [-3, -3, 5]])

H5 = np.array([[-3, -3, -3],
               [-3, 0, 5],
               [-3, 5, 5]])

H6 = np.array([[-3, -3, -3],
               [-3, 0, -3],
               [5, 5, 5]])

H7 = np.array([[-3, -3, -3],
               [5, 0, -3],
               [5, 5, -3]])

H8= np.array([[5, -3, -3],
               [5, 0, -3],
               [5, -3, -3]])

img1 = cv2.imread(filename='XLAandTGMT\\Buoi4\\Picture2.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(filename='XLAandTGMT\\Buoi4\\Picture3.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread(filename='XLAandTGMT\\Buoi4\\Picture1.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(3, 3, figsize=(8, 8))

img_sobel_x = conv(img1, prewitt_Hx)
img_sobel_y =conv(img1, prewitt_Hy)

img_new_1 = conv(img2, H1)
img_new_2 = conv(img2, H2)
img_new_3 = conv(img3, H3)
img_new_4 = conv(img3, H4)
# img_new = conv(img_new, H5)
# img_new = conv(img_new, H6)
# img_new = conv(img_new, H7)
# img_new = conv(img_new, H8)

ax[0][0].imshow(img1, cmap='gray')
ax[0][1].imshow(img_sobel_x, cmap='gray')
ax[0][2].imshow(img_sobel_y, cmap='gray')
ax[1][0].imshow(img2, cmap='gray')
ax[1][1].imshow(img_new_1, cmap='gray')
ax[1][2].imshow(img_new_2, cmap='gray')
ax[2][0].imshow(img3, cmap='gray')
ax[2][1].imshow(img_new_3, cmap='gray')
ax[2][2].imshow(img_new_4, cmap='gray')

plt.show()