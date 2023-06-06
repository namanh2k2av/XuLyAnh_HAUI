import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

def init_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_points_to_clusters(X, centroids):
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    assignment = np.argmin(distances, axis=1)
    return assignment

def update_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for cluster_idx in range(k):
        cluster_points = X[clusters == cluster_idx]
        centroids[cluster_idx] = np.mean(cluster_points, axis=0)
    return centroids

def kmeans(X, k, max_iters=100):
    centroids = init_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_points_to_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids
        
def convert_image(img, k):
    clusters, centroids = kmeans(img, k)
    img_new = []
    for i in range(len(clusters)):
        for j in range(len(centroids)):
            if clusters[i] == j:
                img_new.append(centroids[j])    
    return img_new

img = cv2.imread("XLAandTGMT\\Buoi5\\Picture4.png")
img = np.array(img, dtype=np.float64) / 255
w, h, d = tuple(img.shape)
image_array = np.reshape(img, (w * h, d))
imgs = []
for i in [2, 3, 5, 7, 10]:
    kmean = convert_image(image_array, i)
    new_image = np.reshape(kmean, (w, h, d))
    imgs.append(new_image)
# cv2.imshow('img', img)
# cv2.imshow('img2',im)
# cv2.waitKey(0)
fig, ax = plt.subplots(2, 3, figsize=(10, 6))
ax[0][0].imshow(img,cmap='gray')
ax[0][0].set_title("Original Image")
ax[0][1].imshow(imgs[0])
ax[0][1].set_title("K-Means 2 cluster")
ax[0][2].imshow(imgs[1])
ax[0][2].set_title("K-Means 3 cluster")
ax[1][0].imshow(imgs[2])
ax[1][0].set_title("K-Means 5 cluster")
ax[1][1].imshow(imgs[3])
ax[1][1].set_title("K-Means 7 cluster")
ax[1][2].imshow(imgs[4])
ax[1][2].set_title("K-Means 10 cluster")
plt.show()