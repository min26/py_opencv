"""
$> gh auth login
$> ... ... 
"""
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
        ---- cv2.THRESH_BINARY: if (pixels>threshold) than 255 else 0
        ---- cv2.THRESH_BINARY_INV: opposit case of _BINARY
        ---- cv2.THRESH_TRUNC: if (pixels>threshold) than turncated
        ---- cv2.THRESH_TOZERO:if (pixel<threshold) pixel==0
        ---- cv2.THRESH_TOZERO_INV: opposit case of _TOZERO 
        
        cv2.adaptiveThreshold(src, maxVal, adaptiveMethod, thresholdType,
                                blocksize, constant)
        --adaptiveMethod:
        ---- cv2.ADAPTIVE_THRESH_MEAN_C: 
        ---- cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 
        
        """
        image = cv2.imread("images/dog.jpg")
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ## 1. simple threshold
        # ret, thres1 = cv2.threshold(grayimg, 120, 255, cv2.THRESH_BINARY)
        # ret, thres2 = cv2.threshold(grayimg, 120, 255, cv2.THRESH_TOZERO)
        ## 2. Otsu threshold
        ret, thres3 = cv2.threshold(grayimg, 120, 255, cv2.THRESH_BINARY + 
                                                        cv2.THRESH_OTSU)
        ## 3. adaptive threshold
        # thres1 = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                 cv2.THRESH_BINARY, 199, 5)
        # thres2 = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                 cv2.THRESH_BINARY, 199, 5)
        # cv2.imshow('Binary', thres1)
        # cv2.imshow('Set 0', thres2)
        cv2.imshow('Otus', thres3)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    def subtractBackground(self):
        capt = cv2.VideoCapture('image/walking.mp4')
        ## 1. 
        # fgbg = cv2.createBackgroundSubtractorMOG2()
        # while(1):
        #     ret, frame = capt.read()
        #     fgmask = fgbg.apply(frame)
        #     cv2.imshow('fgmask', fgmask)
        #     cv2.imshow('frame', frame)
        #     #
        #     if cv2.waitKey(30) & 0xff == 27:
        #         break
        ## 2. 
        _, image = capt.read()
        avgVal = np.float32(image)
        while(1):
            _, image = capt.read()
            ## update the running average
            cv2.accumulateWeighted(image, avgVal, 0.02)
            ## convert matrix to absolute value, 8bit
            resultFrame = cv2.convertScaleAbs(avgVal)
            ## or
            ## resultFrame = cv2.RunningAvg(image, acc, alpha)
            ##
            cv2.imshow("input", image)
            cv2.imshow("avgVal", resultFrame)
            ##
            if cv2.waitKey(30) & 0xff == 27:
                break
        capt.release()
        cv2.destroyAllWindows()



        capt.release()
        cv2.destroyAllWindows()




    def colorFilter(self):
        """        
        """
        capture = cv2.VideoCapture(0)
        while(1):
            _,frame = capture.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Threshold of 'blue' in HSV.
            lower_blue = np.array([60,35,140])
            upper_blue = np.array([180, 255, 255])             
            # 
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            #
            cv2.imshow('frime', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('result', result)
            cv2.waitKey(0)
        
        # need somthing to stop here!!
        cv2.destroyAllWindows()
        capture.release()

    def bilateralFilter(self):
        image = cv2.imread("images/elephant.png")    
        ## cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
        ## -- d: diameter of each pixel
        ## -- sigmaColor: greater value, colors farther to each other will mix
        ## -- sigmaSpace: greater value, further pixels will mix, within sigmaColor range
        bilateral = cv2.bilateralFilter(image, 15, 75, 75)   
        cv2.imwrite('bilateral.jpg', bilateral)


    def colorSpace(self):
        image = cv2.imread("images/elephant.png")
        # ## 1. Edge mapping
        # lap = cv2.Laplacian(image, cv2.CV_64F)    
        # cv2.imshow("Edge map", lap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # ## 2. heat map
        # plt.imshow(image, cmap='hot')        
        # plt.show()
        ## 3. spectral map
        plt.imshow(image, cmap='nipy_spectral')
        plt.show()

    ###################
    def contours(self):
        """
        Find co-ordinate of contours
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        img2 = cv2.imread('images/testing.jpg', cv2.IMREAD_COLOR)
        img1 = cv2.imread('images/testing.jpg', cv2.IMREAD_GRAYSCALE)
        ## convert to binary image
        _,threshold = cv2.threshold(img1, 110, 255, cv2.THRESH_BINARY)
        ## detecting contours in image
        contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            ## draws boundary of contours
            cv2.drawContours(img2, [approx], 0, (0,0,255), 5)
            n = approx.ravel()
            i = 0
            for j in n:
                if (i % 2 == 0):
                    x = n[i]
                    y = n[i+1]
                    string = str(x) + " " + str(y)
                    if (i == 0):
                        cv2.putText(img2, "Arrow", (x,y), font, 0.5, (255, 0, 0))
                    else:
                        cv2.putText(img2, string, (x,y), font, 0.5, (0,255, 0))
                i = i + 1
        
        cv2.imshow('img2', img2)
        ##  Exiting with 'q' keyboard. 
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def removeDamage(self):
        os.chdir("./images")
        damaged = cv2.imread("cat_damaged.png")
        height, width = damaged.shape[0], damaged.shape[1]
        #
        ## 1. Mask: covert pixels greater than zero to black, black becomes white
        for i in range(height):
            for j in range(width):
                if damaged[i,j].sum() > 0:
                    damaged[i,j] = 0 #black
                else:
                    damaged[i,j] = [255,255,255] #white 
        cat_mask = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)      
        #cv2.imwrite('cat_mask.jpg', cat_mask)
        #
        ## 2. Inpainted with mask
        original = cv2.imread("cat_damaged.png")
        result = cv2.inpaint(original, cat_mask, 3, cv2.INPAINT_NS)
        # cv2.imwrite('inpained.jpg', result)
        ##
        cv2.imshow("inpainted", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def transformation(self):
        os.chdir("./images")
        img = cv2.imread("elephant.png")
        #
        # ## 1. log transform
        # c = 255 / (np.log(1+ np.max(img)))
        # log_trans = c * np.log(1+ img)
        # result = np.array(log_trans, dtype=np.uint8)
        ## 2. try with gamma value = [0.1, 0.5. 1.2, 2.2]
        gamma = 1.2
        result = np.array(255 * (img /255) ** gamma, dtype = 'uint8')
        #cv2.imwrite('transform.jpg', result)
        cv2.imshow("transform", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()     

    def registration(self):
        img1 = cv2.imread("align1.jpg")
        img2 = cv2.imread("aligh2.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BFR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        height, width = img2.shape
        ## Find keypoints and descriptors
        # The first arg is the image, second arg is the mask
        kp1, d1 = orb_detector, detectAndCompute(img1, None)
        kp2, d2 = orb_detector, detectAndCompute(img2, None)
        # Match features between the two images
        # Brute force matcher with hanning distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        # Match the two sets of descriptors
        matches = matcher.match(d1, d2)
        # Sort matches on the basis of their hanning distance
        matches.sort(key = lambda x: x.distance)
        # Take the top 90% matches
        matches = matches[:int(len(matches)* 0.9)]
        no_of_matches = len(matches)
        ##
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))
        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt 
            p2[i, :] = kp2[matches[i].queryIdx].pt
        ## Find the homography matrix
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAS)
        ## 
        trans = cv2.warpPerspective(img1_color, homography, (width, height))
        ##
        cv2.imwrite('output.jpg', trans)

        


if __name__ == "__main__":
    mycv = CV2test()
    #mycv.edgeDetection()
    #mycv.removeDamage()
    mycv.transformation()