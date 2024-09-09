"""
OpenCV with Anaconda package
install Anaconda
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # 1. read/write 
#
# # cv2.imread(path, flag)
# # - flag:   cv2.IMREAD_COLOR (default)
# #           cv2.IMREAD_GRAYSCALE
# #           cv2.IMREAD_UNCHANGED
# image = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
# # 
# cv2.imwrite('graydog.png', image)

# # 2. show image and key wait
#
# image = cv2.imread("dog.jpg")
# print (image.shape)
# # cv2.imshow('title', image)
# cv2.imshow("Dog", image)
# # cv2.waitkey(milliseconds) 0 = until close
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 2.1. image path
# impath = '/home/min/Document/py_opendv/dog.jpg'
# image = cv2.imread(path, 0) ## 0 = cv2.IMREAD_GRAYSCALE
# cv2.imshow("Dog", image)


# 3. convert colors
# 
image = cv2.imread("dog.jpg")
# # Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #
plt.imshow(rgb_image)
plt.waitforbuttonpress()
plt.close('all')




# img = cv2.imread("dog.jpg")
# img[0,0] = [255, 255, 255]
# imshow()





