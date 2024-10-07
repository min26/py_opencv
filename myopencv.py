"""
OpenCV with Anaconda package
install Anaconda
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CV2test:
    def __init__(self):
        pass

    def readwrite(self):
        """
        --- 1. read image --------------------
        # cv2.imread(path, <flag>)
        # - 1.1. flag
        # cv2.IMREAD_COLOR (default)
        # cv2.IMREAD_GRAYSCALE
        # cv2.IMREAD_UNCHANGED
        # - 1.1. path
        impath = '/home/min/Document/py_opendv/dog.jpg'
        grayimage = cv2.imread(impath, 0) ## 0 = cv2.IMREAD_GRAYSCALE
        grayimage = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
        """
        image = cv2.imread("images/dog.jpg") ## cv2.IMAGE_COLOR
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

    def arithmetics(self):
        """
        # 1. Adding pixels
        # cv2.add(img1, img2)
        # cv2.addWeighted(img1,weight1, img2,weight2, gammaValue)
        # --gammaValue: meaurement of light
        newimg = cv2.addWeighted(img1,0.5,img2,0.3, 0)
        
        # 2. subtract pixels
        newimg = cv2.subtract(img1, img2)
        
        # 3. bitwise operation
        # cv2.bitwise_[and/or/xor/not](src1, src2, dest, mask=None)
        # --mask: 8bit mask
        """
        img1 = cv2.imread("images/building.jpg")
        img2 = cv2.imread("images/space.jpg")
        #
        newimg = cv2.bitwise_and(img1,img2,mask=None)        
        #
        cv2.imshow('Arithmetics', newimg)   
        ## deallocate any associated memory usage
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    def colorspace(self):
        ## ---5. Color space --------------
        ## RGB, CMYK(Cyan, Magenta, Yellow, Black), HSV(Hue, Saturation, Value)
        color_image = cv2.imread("images/dog.jpg")
        ## split by color
        B,G,R = cv2.split(color_image)
        #
        cv2.imshow("blue", B)
        cv2.waitKey(0)
        cv2.imshow("red", R)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def imgResize(self):
        """
        OpenCV provides interpolation resize methods
        cv2.resize(source, dsize, dest, fx, fy, interpolation)
        -- source: single channel, 8bit float
        -- dsize: output size array (x,y), optional
        -- fx/fy: scale factor, optional
        -- interpolation, optional: 
        ---- cv2.INTER_AREA: for shrink an image
        ---- cv2.INTER_CUBIC: bicubic, slow but efficient
        ---- cv2.INTER_NEAREST: nearest neighbor
        ---- cv2.INTER_LINEAR: primarily zooming. default. 
        """
        image = cv2.imread("images/dog.jpg")
        ## get image dimensions
        h,w = image.shape[:2]
        aspect_ratio = h/w
        ## desired width
        new_width = 800         
        new_height = int(new_width * aspect_ratio)
        ##
        ratio = cv2.resize(image, (new_width, new_height))
        half = cv2.resize(image, (0,0), fx=0.1, fy=0.1)        
        stretch = cv2.resize(image, (780,540),interpolation=cv2.INTER_LINEAR)
        titles=["origin","ratio","half","stretch"]
        images=[image, ratio, half, stretch]
        ##
        for i in range(4):
            plt.subplot(2,2, i+1)
            plt.title(titles[i])
            plt.imshow(images[i])
        plt.show()


if __name__ == "__main__":
    mycv = CV2test()
    mycv.imgResize()

