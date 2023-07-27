import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import time
import math
import PIL


def showImage(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawKeypoints(img, kp, _showImage=False):
    #imgWithKeypoints = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    imgWithKeypoints = cv2.drawKeypoints(img, kp, None, flags=0)
    if _showImage:
        showImage(imgWithKeypoints)
    return imgWithKeypoints

def fastFeatureDetector(img, title='FAST Feature Detector', showImg=False, 
                        threshold = 10, type = 2, nonmaxSuppression = True):
    '''
    FAST Tutorial: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_fast.html
    '''
    fast = cv2.FastFeatureDetector_create(threshold=threshold, type=type, nonmaxSuppression=nonmaxSuppression)
    keypoints = fast.detect(img, None)
    imgWithKeypoints = cv2.drawKeypoints(img, keypoints, None)

    print("Threshold: ", fast.getThreshold())
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print("neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(keypoints))

    if showImg: showImage(imgWithKeypoints, title)
    return keypoints

def orbDetector(img, n):
    orb = cv2.ORB_create(n)
    kp = orb.detect(img, None)
    return kp

def orbDetectorAndDescriptor(img, n, showImage = False):
    orb = cv2.ORB_create(n)
    kp, des = orb.detectAndCompute(img, None)
    if showImage:
        imgWithKeypoints = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        plt.imshow(imgWithKeypoints), plt.show()
    return kp, des

def orbImgWithKeypoints(img, n):
    '''
        Function created only to test the keypoints distribution
        after applying image segmentation.
    '''
    orb = cv2.ORB_create(n)
    kp, des = orb.detectAndCompute(img, None)
    imgWithKeypoints = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    return imgWithKeypoints

def imgSegmentation(img, nBlocks=(2,2)):
    horizontal = np.array_split(img, nBlocks[0])
    splitted_img = [np.array_split(block, nBlocks[1], axis=1) for block in horizontal]
   
    minHeigh, minWidth = 0, 0
    heighs, widths = [], []

    for row in range(nBlocks[0]):
        heighs.append(minHeigh)
        minHeigh += len(splitted_img[row][0])

    for col in range(nBlocks[1]):
        widths.append(minWidth)
        minWidth += len(splitted_img[0][col][0])

    minPoints = np.vstack((heighs, widths))
    return np.asarray(splitted_img, dtype=np.ndarray).reshape(nBlocks), minPoints

def revertImageSegmentation(imgArray, nBlocks=(2,2), title='Merged Image'):
    for h in range(nBlocks[0]):
        buffer = imgArray[h, 0]
        for w in range(1, nBlocks[1]):
            print(f"Img: {imgArray[h, w].shape} - Buffer: {buffer.shape}")
            buffer = np.hstack((buffer, imgArray[h, w]))
        if h == 0: result = buffer
        else: result = np.vstack((result, buffer))
    showImage(result, title = title)

def segmentedOrb(img, numFeatures, nBlocks=(2,2)):
    imgs, segmentationPoints = imgSegmentation(img, nBlocks=nBlocks)
    h, w = segmentationPoints
    result = []
    for row in range(imgs.shape[0]):
        for col in range(imgs.shape[1]):
            kp = orbDetector(imgs[row][col], numFeatures)
            print(row, col, len(kp))
            for p in kp:
                p.pt = (p.pt[0]+w[col], p.pt[1]+h[row])
            if len(result): result = np.hstack((result, kp))
            else: result = kp
    return result

def applyLoG(img, blurWindowSize=(3, 3), type=cv2.CV_8U):
    blur = cv2.GaussianBlur(img, blurWindowSize, 0)
    laplacian = cv2.Laplacian(blur, type)
    return laplacian

def returnKeyPointArray(xCoordinates, yCoordinates, size=1):    # TODO
    kp = []
    if isinstance(xCoordinates[0], float):
        for _ in range(len(xCoordinates)):
            kp.append(cv2.KeyPoint(x=xCoordinates[_], y=yCoordinates[_], size=size))
    else:
        for _ in range(len(xCoordinates)):
            kp.append(cv2.KeyPoint(x=float(xCoordinates[_]), y=float(yCoordinates[_]), size=size))
    return kp        

def logDetector(img, blurWindowSize=(3, 3), _showImage=False, type=cv2.CV_8U):
    img = applyLoG(img, blurWindowSize, type=type)
    if _showImage:
        showImage(img)
    kp_zeros = np.argwhere(img>=10)
    kp_x, kp_y = kp_zeros[:, 1], kp_zeros[:, 0] # colunas, linhas
    kp = returnKeyPointArray(kp_x, kp_y)  
    return kp, drawKeypoints(img, kp=kp, _showImage=True)


def cannyEdgeDetection(img, title = 'Canny Edge Detector', showImg = True):     # TODO
    # https://www.youtube.com/watch?v=hUC1uoigH6s&list=PL2zRqk16wsdqXEMpHrc4Qnb5rA1Cylrhx&index=5
    pass

