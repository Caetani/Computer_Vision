import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


def showImage(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scaledDepthMap(depth):
    maxVal, minVal = np.max(depth), np.min(depth)
    fig, ax = plt.subplots()
    h = ax.imshow(depth, cmap = 'Reds')
    fig.colorbar(h)
    plt.show()

def drawKeypoints(img, kp, _showImage=False):
    imgWithKeypoints = cv2.drawKeypoints(img, kp, None, flags=0)
    if _showImage:
        showImage(imgWithKeypoints)
    return imgWithKeypoints

def fastFeatureDetector(img, title='FAST Feature Detector', showImg=False, 
                        threshold=10, type=2, nonmaxSuppression=True):

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

def orbDescriptor(img, kp):
    orb = cv2.ORB_create()
    kp, des = orb.compute(img, kp)
    return kp, des

def orbDetectorAndDescriptor(img, n, showImage=False):
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

def applyLoG(img, blurWindowSize=(3, 3), type=cv2.CV_8U, _sigma=4):
    blur = cv2.GaussianBlur(img, blurWindowSize, sigmaX=_sigma, sigmaY=_sigma)
    laplacian = cv2.Laplacian(blur, type)
    return laplacian

def returnKeyPointArray(coordinatesArray, size=1):    # TODO
    keyPoints = []
    for x, y in coordinatesArray:
        keyPoints.append(cv2.KeyPoint(x=x, y=y, size=size))
    return keyPoints      

def logDetector(img, blurWindowSize=(3, 3), _showImage=False, type=cv2.CV_8U, sigma=2):
    img = applyLoG(img, blurWindowSize, type=type, _sigma=sigma)
    if _showImage:
        showImage(img)
    kp_zeros = np.argwhere(img>=10)
    kp_x, kp_y = kp_zeros[:, 1], kp_zeros[:, 0] # colunas, linhas
    kp = returnKeyPointArray(kp_x, kp_y)  
    return kp, drawKeypoints(img, kp=kp, _showImage=_showImage)

def getStereoImages(id, cameraMatrix, distCoeffs):
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
        
    assert imgL.shape == imgR.shape, "Images sizes do not match."
    return imgL, imgR, imgL.shape # shape = (height, width)

def resizeStereoImages(imgL, imgR, ratio, interpol=cv2.INTER_AREA):
    assert imgL.shape == imgR.shape, "Images sizes do not match."
    height, width = int(imgL.shape[0]*ratio), int(imgL.shape[1]*ratio)
    imgL = cv2.resize(imgL, (width, height), interpolation = cv2.INTER_AREA)
    imgR = cv2.resize(imgR, (width, height), interpolation = cv2.INTER_AREA)
    return imgL, imgR

def resizeImage(img, ratio, interpol=cv2.INTER_AREA):
    height, width = int(img.shape[0]*ratio), int(img.shape[1]*ratio)
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return img

def getMatchesCoordinates(matches, leftKeypoints, rightKeypoints):
    leftCoordinates, rightCoordinates = [], []
    for match in matches:
        leftCoordinates.append(leftKeypoints[match.queryIdx].pt)
        rightCoordinates.append(rightKeypoints[match.trainIdx].pt)
    return np.array(leftCoordinates), np.array(rightCoordinates)

def inliersRatio(inliersArray):
    return np.count_nonzero(inliersArray) / len(inliersArray)

def cannyEdgeDetection(img, title = 'Canny Edge Detector', showImg = True):     # TODO
    # https://www.youtube.com/watch?v=hUC1uoigH6s&list=PL2zRqk16wsdqXEMpHrc4Qnb5rA1Cylrhx&index=5
    pass

