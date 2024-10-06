"""
OpenCV with Anaconda package
install Anaconda
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. read image --------------------
## cv2.imread(path, <flag>)
## - 1.1. flag
## cv2.IMREAD_COLOR (default)
## cv2.IMREAD_GRAYSCALE
## cv2.IMREAD_UNCHANGED
## - 1.1. path
# impath = '/home/min/Document/py_opendv/dog.jpg'
# grayimage = cv2.imread(impath, 0) ## 0 = cv2.IMREAD_GRAYSCALE
# grayimage = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("dog.jpg") ## cv2.IMAGE_COLOR
print (image.shape)
#
# --- 2. convert image --------------------
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# --- 3. write image -----------
## support format: jpeg, png, tiff, bmp, ppm, pgm
new_dir = '/home/min/Documents'
os.chdir(new_dir)
success = cv2.imwrite('graydog.png', gray_image)
if not success:
    print("ERROR write image")
else:
    print(os.listdir(new_dir))
# 

#
# --- 4. show image ----------------------
# - 4.1. cv2.imshow('title', image)
cv2.imshow("Dog", gray_image)
cv2.waitKey(0) ## milliseconds
cv2.destroyAllWindows()
#
# - 4.2. matplotlib.imshow(image)
# plt.imshow(gray_image)
# plt.waitforbuttonpress()
# plt.close('all')


# img = cv2.imread("dog.jpg")
# img[0,0] = [255, 255, 255]
# imshow()





