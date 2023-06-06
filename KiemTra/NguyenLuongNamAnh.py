#Đề 1
import numpy as np
import cv2


img = cv2.imread('XLAandTGMT\\KiemTra\\anh1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img[img>100] = 255
img[img<=100] = 0

kernel_1 = np.array([[0, 0, 1],
                     [0, 1, 1],
                     [0, 0, 0]])

kernel_2 = np.array([[0, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0]])


kernel_3 = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]])

kernel_4 = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])

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

img_new = dilation(img, kernel_3)
img_new = dilation(img_new, kernel_3)
img_new = dilation(img_new, kernel_3)
img_new = dilation(img_new, kernel_4)
img_new = erosion(img_new, kernel_3)
img_new = dilation(img_new, kernel_3)
img_new = erosion(img_new, kernel_3)
img_new = erosion(img_new, kernel_3)
img_new = erosion(img_new, kernel_4)
img_new = erosion(img_new, kernel_3)
cv2.imshow('img',img)
cv2.imshow('Result', img_new)
cv2.waitKey(0)