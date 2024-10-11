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
        ##
        ## creating a histograms equalization
        # equ = cv2.equalizeHist(img)
        # res = np.hstack((img, equ))
        # cv2.imshow("Equalize", res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ##
        ## calculate pixels in rage 0-255
        histr = cv2.calcHist([img],[0],None,[256],[0,256])        
        ##
        plt.plot(histr)
        plt.show()
        ##

    def thresholding(self):
        """
        cv2.threshold(src, thresholdVal, maxVal, thresholdTech)
        --thresholdTech:
        ---- cv2.THRESH_BINARY: inverted or opposit
        ---- cv2.THRESH_TRUNC: if (pixels>threshold) than turncated
        ---- cv2.THRESH_TOZERO:if (pixel<threshold) pixel==0
        """

if __name__ == "__main__":
    mycv = CV2test()
    #mycv.edgeDetection()
    mycv.imgHistogram()