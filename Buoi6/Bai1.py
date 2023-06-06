import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

def Hist(image, bins):
    hist1 = np.zeros(256, dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist1[image[i,j]] += 1
    hist2 = []
    i = 0
    k = 256 / bins
    while i < len(hist1):
        temp = np.zeros(4)
        temp = hist1[i : i + int(k)]
        hist2.append(temp.min())
        i += int(k)    
    return hist2

def calcHist(image,bins=8):
    hist = np.zeros((bins*3), dtype=np.float32)
    pixel_count = 0
    b, g, r = cv2.split(image)
    for channel in (b, g, r):
        channel_hist = Hist(channel, bins)
        hist[pixel_count : pixel_count + bins] = channel_hist
        pixel_count += bins
    return hist

test_img = cv2.imread('XLAandTGMT\\Buoi6\\test\\test\\t3.jpg')
test_img = cv2.resize(test_img, (300, 300))

data_imgs = []
data_hists = []

for i in range(1, 12):
    data_img = cv2.imread('XLAandTGMT\\Buoi6\\data\\data\\a{}.jpg'.format(i))
    data_img = cv2.resize(data_img, (300, 300))
    hist = calcHist(data_img)
    hist = cv2.normalize(hist, hist)
    data_imgs.append(data_img)
    data_hists.append(hist)
    
for i in range(1, 6):
    data_img = cv2.imread('XLAandTGMT\\Buoi6\\data\\data\\c{}.jpg'.format(i))
    data_img = cv2.resize(data_img, (300, 300))
    hist = calcHist(data_img)
    hist = cv2.normalize(hist, hist)
    data_imgs.append(data_img)
    data_hists.append(hist)

test_hist = calcHist(test_img)
test_hist = cv2.normalize(test_hist, test_hist)

K = 3
neigh = NearestNeighbors(n_neighbors=K)
neigh.fit(data_hists)
distances, indices = neigh.kneighbors([test_hist])

print(distances)
print(indices)

cv2.imshow('Test image', test_img)
for i in range(K):
    data_img = data_imgs[indices[0][i]]
    cv2.imshow('Data image {}'.format(i+1), data_img)
cv2.waitKey(0)
cv2.destroyAllWindows()