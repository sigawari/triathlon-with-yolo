import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

def calibrate(showPics=True):
    # Read image
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages//calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, "*.jpg"))

    #initialize
    nRows = 9
    nCols = 16
    termCriteria = (cv.TERM_CRITERA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = []

    #find corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

        if cornersFound == True:
            worldPtsList.append(worldPltsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11,11), (-1,-1), termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitkey(500)
    cv.destroyAllWindows()

    #Calibrate
    repError,camMatrix,distCoeff,rvecs,tvecs = cv.calibrateCamera
    (worldPtsList, imgPtsList, imgGray.shape[::-1],None,None)
    print('Camera matrix: \n',camMatrix)
    print("Reproc Error (pixels): {:.4f}".format(repError))

    #Save calibration parameters
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(paramPath,
             repError = repError,
             camMatrix = camMatrix,
             distCoeff=distCoeff,
             rvecs=rvecs,
             tvecs=tvecs)
    
    return camMatrix,distCoeff

def removeDistortion(camMatrix, distCoeff):
    root = os.getcwd()
    imgPath = os.path.join(root, 'demoImages//distortion2.jpg')
    img = cv.imread(imgPath)
    height,width = img.shape[:2]
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff,(width,height),1,(width,height))
    imgUndist = cv.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

    # Draw line to see distortion change
    cv.line(img, (1769, 103), (1769, 900), (0, 0, 255), 5)
    cv.line(imgUndist, (1769, 103), (1769, 900), (0, 255, 0), 5)

    # Display the original and undistorted images
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runCalibration():
    calibrate(showPics=True)

def runRemoveDistortion():
    camMatrix,distCoeff = calibrate(showPics=False)
    removeDistortion(camMatrix,distCoeff)

if __name__ == '__main__':
    # runCalibration()
    runRemoveDistortion()