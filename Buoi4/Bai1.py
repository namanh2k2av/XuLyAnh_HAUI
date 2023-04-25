import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(filename='XLAandTGMT\\Buoi4\\Picture1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(2, 1, figsize=(5, 5))

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

img_new = nguongtudong(img)

ax[0].imshow(img, cmap='gray')
ax[1].imshow(img_new, cmap='gray')
plt.show()