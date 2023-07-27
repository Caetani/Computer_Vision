import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from commom import * 

width, height = 4032, 3024
imgShape = (height, width)
calibrationData = pd.read_pickle("Camera Parameters/calibration.pkl")
cameraMatrix, distCoeffs = calibrationData[0], calibrationData[1][0]
stereoImgSelection = {
    0: 'final_',
    1: 'corr_',
    2: 'small_'
}
imgPrefix = stereoImgSelection[0]
imgL = cv2.imread("Stereo Images/" + imgPrefix + 'left.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("Stereo Images/" + imgPrefix + 'right.jpg', cv2.IMREAD_GRAYSCALE)
imgL = cv2.undistort(imgL, cameraMatrix, distCoeffs, None)
imgR = cv2.undistort(imgR, cameraMatrix, distCoeffs, None)

print(imgL.shape)

imgPercentage = 0.3
width, height = int(width*imgPercentage), int(height*imgPercentage)
imgL = cv2.resize(imgL, (width, height), interpolation = cv2.INTER_AREA)
imgR = cv2.resize(imgR, (width, height), interpolation = cv2.INTER_AREA)

#######################################################

#kp = fastFeatureDetector(imgL, threshold=5, type=2, showImg=True)
#kp, des = orbDetector(imgL, 500)
n = 10_000
segmentationBlocks = (2, 2)

kp = segmentedOrb(imgL, n, nBlocks=segmentationBlocks)
drawKeypoints(imgL, kp)

############# My part of the code above ############

