import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CV2test:
    def __init__(self):
        pass

    def imgHistogram(self):
        """
        matplotlib supports only PNG images.
        cv2.calcHist(src, channel, mask, histSize, ranges, [hist], [accumulate])
        -- channel: 0 for gray; 0,1 or 2 for color channels
        -- mask = None
        -- histSize = 256 (full scale)
        -- ranges = [0,256]
        """
        img = cv2.imread("images/elephant.png", cv2.IMREAD_GRAYSCALE)
        ## calculate pixels in rage 0-255
        histr = cv2.calcHist([img],[0],None,[256],[0,256])
        ##
        plt.plot(histr)
        plt.show()



if __name__ == "__main__":
    mycv = CV2test()
    #mycv.edgeDetection()
    mycv.imgHistogram()