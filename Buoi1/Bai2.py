import numpy as np
import matplotlib.pyplot as plt
import cv2

def empty(a):
    pass

cv2.namedWindow('Do sang')
cv2.resizeWindow('Do sang', 300, 50)
cv2.createTrackbar('Value', 'Do sang', 0, 255, empty)

def tanggiam(img, c):
    temp = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp[i,j] += c
            if img[i,j] <= 0:
                temp[i,j] = 0
            elif img[i,j] >= 255:
                temp[i,j] = 255
    return temp

while True:
    c = cv2.getTrackbarPos('Value', 'Do sang')
    img = cv2.imread('XLAandTGMT\\Buoi1\\anh3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Origin', img)
    img_new = img.copy()
    img_new = tanggiam(img_new, c)
    print(c)
    cv2.imshow('Result', img_new)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break