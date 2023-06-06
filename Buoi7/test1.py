import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


def resize_image(image, new_size):
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def calculate_lbp(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    lbp = np.zeros_like(gray)
    height, width = gray.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center = gray[i, j]
            code = 0
            code |= (gray[i - 1, j - 1] > center) << 7
            code |= (gray[i - 1, j] > center) << 6
            code |= (gray[i - 1, j + 1] > center) << 5
            code |= (gray[i, j + 1] > center) << 4
            code |= (gray[i + 1, j + 1] > center) << 3
            code |= (gray[i + 1, j] > center) << 2
            code |= (gray[i + 1, j - 1] > center) << 1
            code |= (gray[i, j - 1] > center) << 0
            lbp[i, j] = code

    return lbp

def search_similar_images(test_image, train_images, k):
    test_lbp = calculate_lbp(test_image)
    train_lbps = [calculate_lbp(image) for image in train_images]

    train_lbps = np.array(train_lbps).reshape(len(train_lbps), -1)
    test_lbp = test_lbp.reshape(1, -1)

    neigh = NearestNeighbors(n_neighbors=k, metric='hamming')
    neigh.fit(train_lbps)

    distances, indices = neigh.kneighbors(test_lbp)

    return indices[0]


# Thông tin về bộ dữ liệu và ảnh test
train_image_paths = ["train1.png", "train2.png", "train3.bmp", "train4.bmp", "train5.bmp", "train6.bmp", "train7.bmp",
                     "train8.bmp", "train9.bmp", "train10.bmp","train3_1.bmp","train3_2.bmp","train3_3.bmp","train3_4.bmp"]  # Đường dẫn đến các ảnh train
test_image_path = "XLAandTGMT\\Buoi7\\patterm\\patterm\\test3.bmp"  # Đường dẫn đến ảnh test

# Đọc ảnh train và ảnh test
train_images = []
for path in train_image_paths:
    image = cv2.imread("XLAandTGMT\\Buoi7\\patterm\\patterm\\"+path)
    if image is not None:
        train_images.append(image)

test_image = cv2.imread(test_image_path)
if test_image is None:
    print("Không thể đọc ảnh test")
    exit()

# Chuyển tất cả ảnh train và test về cùng kích thước
new_size = (200, 200)
train_images = [resize_image(image, new_size) for image in train_images]
test_image = resize_image(test_image, new_size)

# Tìm kiếm ảnh tương tự
k = 5  # Số lượng ảnh train
# Tìm kiếm ảnh tương tự
similar_indices = search_similar_images(test_image, train_images, k)

# Hiển thị ảnh test và các ảnh train tìm được
cv2.imshow("Test Image", test_image)
for i in range(k):
    similar_image = train_images[similar_indices[i]]
    cv2.imshow(f"Similar Image {i + 1}", similar_image)

cv2.waitKey(0)
cv2.destroyAllWindows()