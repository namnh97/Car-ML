# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os.path as path
import pickle
import numpy as np
import cv2
import glob 
import matplotlib.pyplot as plt

calib_images_dir = "camera_cal"
cx = 9
cy = 6

def check_calibrate(calibration_cache):
    if path.exists(calibration_cache):
        print('Loading cached camera calibration...', end = ' ')
        with open(calibration_cache, 'rb') as dump_file:
            calibration = pickle.load(dump_file)
    else:
        print('Computing camera calibration...', end = ' ')
        calibration = findImgObjPoints(calib_images_dir)
        with open(calibration_cache, 'wb') as dump_file:
            pickle.dump(calibration, dump_file)
    print('Done.')
    
    return calibration

def findChessboardCorners(img, nx, ny):
    """
    Finds the chessboard corners of the supplied image (must be grayscale)
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    return cv2.findChessboardCorners(img, (nx, ny), None)

#find opts, ipts for analyzing
def findImgObjPoints(imgs_paths, nx = 6, ny = 9):
    """
    Returns the objects and image points computed for a set of chessboard pictures taken from the same camera
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    objpts = []
    imgpts = []
    
    # Pre-compute what our object points in the real world should be (the z dimension is 0 as we assume a flat surface)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    imgs_paths = glob.glob(path.join(calib_images_dir, 'calibration*.jpg'))

    for img_path in imgs_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = findChessboardCorners(gray, nx, ny)
        
        if ret:
            # Found the corners of an image
            imgpts.append(corners)
            # Add the same object point since they don't change in the real world
            objpts.append(objp)
    
    return objpts, imgpts

#function undistort_image
def undistort_image(img, objpts, imgpts):
    """
    Returns an undistorted image
    The desired object and image points must also be supplied to this function
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def calibrate(img):
    calibration_cache = 'camera_cal/calibration_data.pickle'
    opts, ipts = check_calibrate(calibration_cache)
    undistored_img = undistort_image(img, opts, ipts)    
    
    return undistored_img

if __name__ == '__main__':
    calibration_cache = 'camera_cal/calibration_data.pickle'

    opts, ipts = check_calibrate(calibration_cache)
        
    img = cv2.imread('input/inputDetectLine.png')
    undistored_img = undistort_image(img, opts, ipts)    
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 7))
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title("original")
    
    ax[1].imshow(undistored_img)
    ax[1].axis("off")
    ax[1].set_title("processed image")
    
    plt.show()