import numpy as np
import matplotlib.pyplot as plt
import cv2

def histo(img):
    his = np.zeros(256, dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i,j]] += 1
    return his

def canbanghis(img, mucxam):
    g, hG = np.unique(img, return_counts=True)
    img_new = img.copy()
    g_new = []
    nxm = img.shape[0] * img.shape[1]
    tG = [hG[0]]
    for i in range(1,len(hG)):
        temp = hG[i] + tG[i-1]
        tG.append(temp)
    for i in range(len(hG)):
        temp = (tG[i] / nxm ) * mucxam - 1
        g_new.append(temp)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(len(g)):
                if img[i,j] == g[k]:
                    img_new[i,j] = g_new[k]               
    return img_new


img = cv2.imread('XLAandTGMT\\Buoi1\\anh2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_new = img.copy()
img_new = canbanghis(img_new, 255)

his = histo(img)
his_new = histo(img_new)

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0][0].imshow(img, cmap='gray')
ax[1][0].plot(his)
ax[0][1].imshow(img_new, cmap='gray')
ax[1][1].plot(his_new)
plt.show()