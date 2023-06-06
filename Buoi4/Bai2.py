import cv2
import numpy as np
import matplotlib.pyplot as plt
# tim bien
def conv(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

sobel_Hx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

sobel_Hy = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])



img1 = cv2.imread(filename='XLAandTGMT\\Buoi4\\Picture2.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(filename='XLAandTGMT\\Buoi4\\Picture3.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.imread(filename='XLAandTGMT\\Buoi4\\Picture1.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(1, 2, figsize=(8, 8))

img_sobel_x = conv(img2, sobel_Hx)
img_sobel_y = conv(img2, sobel_Hy)

edge_img = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))
edge_img = edge_img / np.max(edge_img) * 255
edge_img[edge_img > 0] = 255

# sob_out = ((sob_out / np.max(sob_out)) * 255)

cv2.imshow('Img',img2)
cv2.imshow('Result1',edge_img)

cv2.waitKey(0)
cv2.destroyAllWindows()