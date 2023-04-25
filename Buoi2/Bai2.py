import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loc trung vi

img = cv2.imread('XLAandTGMT\\Buoi2\\Picture4.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(2, 1, figsize=(5, 5))

def median_filter(img, beta):
    img_new = np.copy(img)
    n = img_new.shape[0]
    m = img_new.shape[1]
    for i in range(1, n-1):
        for j in range(1, m-1):
            arr = []
            center_pixel = [i, j]
            xRows = [0, 0, 1, -1, 1, -1, 1, -1]
            yCols = [-1, 1, 0, 0, -1, -1, 1, 1]
            for k in range(8):
                item = [xRows[k], yCols[k]]
                pixel_img_x = center_pixel[0] + item[0]
                pixel_img_y = center_pixel[1] + item[1]
                arr.append(img[pixel_img_x][pixel_img_y])
            arr.append(img[i][j])
            arr = sorted(arr[:])
            temp = arr[len(arr)//2]
            if abs(int(img[i, j]) - int(temp)) <= beta:
                img_new[i, j] = img[i, j]
            else:
                img_new[i, j] = temp
    return img_new
      
img_new = median_filter(img, 3)
for i in range(0,3):
    img_new = median_filter(img_new, 3)

ax[0].imshow(img, cmap='gray')
ax[1].imshow(img_new, cmap='gray')


plt.show()