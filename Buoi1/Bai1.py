import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('XLAandTGMT\\Buoi1\\anh1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(2, 1, figsize=(5, 5))

def histo(img):
    his = np.zeros(256, dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i,j]] += 1
    return his

his = histo(img)

ax[0].imshow(img, cmap='gray')
ax[1].plot(his)

plt.show()