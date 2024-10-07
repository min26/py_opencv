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
        """
        image = cv2.imread("images/dog.jpg") ## cv2.IMAGE_COLOR
        print (image.shape)
        #
        # --- 2. convert image --------------------
        # -- gray image#2
        # cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # -- rgb image
        cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # --- 3. write image -----------
        ## support format: jpeg, png, tiff, bmp, ppm, pgm
        new_dir = '/home/min/Documents'
        os.chdir(new_dir)
        success = cv2.imwrite('graydog.png', cvt_image)
        if not success:
            print("ERROR write image")
        else:
            print(os.listdir(new_dir))
        # 
        # --- 4. show image ----------------------
        # -- 4.1. cv2.imshow('title', image)
        cv2.imshow("Convert", cvt_image)
        cv2.waitKey(0) ## milliseconds
        cv2.destroyAllWindows()
        #
        # -- 4.2. matplotlib.imshow(image)
        # plt.imshow(gray_image)
        # plt.waitforbuttonpress()
        # plt.close('all')

    def gray_image(self):
        ## - 1. read as gray
        # grayimage = cv2.imread(impath, 0) ## 0 = cv2.IMREAD_GRAYSCALE
        # grayimage = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
        #        
        image = cv2.imread("image/dog.jpg")
        ## - 2. convert as gray
        # grayimage = cv2.cvtColor(image, cv2.COLOR_COLOR_BGR2GRAY)
        #
        ## - 3. pixel manipulation
        row, col = image.shape[0:2]
        ## BGR channel to convert 'colored' to 'gray'
        for i in range(row):
            for j in range(col):
                grayimage[i, j] = sum(image[i,j]) * 0.33
        ###
        cv2.imshow('Grayscale',grayimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        image = cv2.imread("images/elephant.png")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w = image_rgb.shape[:2]
        ## 1. Change dimensions with same ratio        
        #aspect_ratio = h/w
        #new_width = 800         
        #new_height = int(new_width * aspect_ratio)
        #
        ## 2. Change with scale factor
        scale1 = 3.0    #increase 3x
        scale2 = 1/3.0  #decrease 3x
        new_width = int(w * scale1)
        new_height = int(h * scale1)
        ##
        scale = cv2.resize(image_rgb, 
                            dsize=(new_width, new_height),
                            interpolation=cv2.INTER_AREA)
        zoom = cv2.resize(image_rgb, 
                            dsize=(new_width, new_height),
                            interpolation=cv2.INTER_CUBIC)
        half = cv2.resize(image_rgb, (0,0), fx=0.1, fy=0.1)        
        #stretch = cv2.resize(image_rgb, (780,540),interpolation=cv2.INTER_LINEAR)
        #
        titles=["origin","scale","zoom","half"]
        images=[image_rgb, scale, zoom, half]
        ##
        for i in range(4):
            plt.subplot(2,2, i+1)
            plt.title(titles[i])
            plt.imshow(images[i])
        plt.tight_layout()
        plt.show()

    def imgRotation(self):
        """
        cv2.getRotationMatrix2D()
        cv2.warpAffine()
        """
        image = cv2.imread("images/elephant.png")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height,width = image_rgb.shape[:2]
        center = (image_rgb.shape[1]/2, image_rgb.shape[0]/2)
        ## rotation 
        # angle = 45
        # scale = 1
        # rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # new_image = cv2.warpAffine(image_rgb, rot_matrix, 
        #                 (image_rgb.shape[1], image_rgb.shape[0]))
        ## translation
        tx = 100
        ty = 100
        tran_matrix = np.array([[1,0,tx], [0,1,ty]],dtype=np.float32)
        new_image = cv2.warpAffine(image_rgb, tran_matrix, (width, height)) 
        ##
        cv2.imshow("Rotation", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def imgErode(self):
        """
        [erode] reduce,...
        cv2.erode(src, kernel, [dst], [anchor], [interations], 
                    [borderType], [boarderValue])
        -- src: input image
        -- kernel: structuring element used for erosion
        -- dst: output image
        -- anchor: default point is kernel center(-1, -1)
        -- iterations: number of times erosion is applied
        -- borderType: cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT,...
        -- borderValue: returns an images
        - returns an output image
        """
        path = "images/dog.jpg"
        image = cv2.imread(path)
        ## 1. erodsion
        kernel = np.ones((5,5), np.uint8)
        new_image = cv2.erode(image, kernel)
        # new_image = cv2.erode(image, kernel, cv2.BORDER_REFLECT)
        #
        ## 2. Make border 
        ## copyMakeBorder(src, top,bottom,left,right, borderType, value)
        ## -- borderType: cv2.BOARDER_DEFAULT, _CONSTANT, _REFLECT
        ## -- value (optional): cv2.BORDER_CONSTANT
        # new_image = cv2.copyMakeBorder(new_image, 5,5,5,5, cv2.BORDER_CONSTANT)
        new_image = cv2.copyMakeBorder(new_image, 100,100,100,100, cv2.BORDER_REFLECT)
        #
        cv2.imshow("Erosion", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def imgBlur(self):
        """
        [blurring]
        -- remove noise, smoothing image, hiding details
        + filter2D(src, ddepth, kernel)
        + blur(src, shapeOfKernel)
        + GaussianBlur(src,shapeOfKernel, sigmaX): reduce image noise and detail
        + medianBlur(src, kernelSize): non-linear digital filter, best to remove salt and pepper noise.
        + bilateralFilter(src, diameter, sigmaColor, sigmaSpacce): non-linear, edge-preserving, noise-reducing. 
        Replaces each pixel with average intensity values from nearby pixels.
        --- shpeOfKernel: matrix 3x3, 5x5, ...
        --- kernelSize: positive integer more than 2.
        --- diameter: similar to kernelSize
        --- sigmaColor: number of color range of pixels
        --- sigmaSpace: space between pixels
        """
        path = "images/dog.jpg"
        image = cv2.imread(path)
        kernel_val = np.ones((5,5), np.float32)/25
        ## blurring
        # blurImg = cv2.filter2D(src=image, ddepth=-1, kernel_val)
        # blurImg = cv.blur(image, (5,5))
        # blurImg = cv2.Gaussianblur(image, (7,7), 0)
        blurImg = cv2.medianBlur(image, 9)
        # blurImg = cv2.bilateralFilter(image, 9, 75, 75)
        cv2.imshow('BlurImg', blurImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mycv = CV2test()
    mycv.imgRotation()

