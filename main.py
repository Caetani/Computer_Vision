import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from commom import * 


calibrationData = pd.read_pickle("Camera Parameters/calibration.pkl")
cameraMatrix, distCoeffs = calibrationData[0], calibrationData[1][0]
f = cameraMatrix[0][0]

#b = 0.1 # m
#R = np.identity(3)
#T = np.array([[0], [-b], [0]])

imgL, imgR, imgShape = getStereoImages(0, cameraMatrix, distCoeffs)
height, width = imgShape

pct = 1
imgL, imgR = resizeStereoImages(imgL, imgR, pct)

numFeatures = 10_000
leftKeypoints, leftDescriptor = orbDetectorAndDescriptor(imgL, numFeatures, showImage=False)
rightKeypoints, rightDescriptor = orbDetectorAndDescriptor(imgR, numFeatures, showImage=False)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)                               # Create BFMatcher object
matches = bf.match(leftDescriptor, rightDescriptor)                                 # Match descriptors.
matches = sorted(matches, key = lambda x:x.distance)                                # Sort them in the order of their distance.
numMatches = 1_000
final_matches = matches[:numMatches]                                                # Select first 'n' matches.

leftPoints, rightPoints = getMatchesCoordinates(final_matches, leftKeypoints, rightKeypoints)

externalParameters, inliers = cv2.estimateAffine2D(leftPoints, rightPoints, confidence=0.999, refineIters=1_000, 
                                                    maxIters=10_000, method=cv2.RANSAC, ransacReprojThreshold=0.01*imgL.shape[0])
print(f"Ratio = {inliersRatio(inliers)}")
print(externalParameters)

