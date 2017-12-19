# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 20:39:29 2017

@author: Yang Sabertooth
"""
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import os

from sklearn import svm
from scipy import ndimage
import math
from skimage.feature import hog
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
#from scipy.misc import imread
from skimage.io import imread


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""

    byteorder = '>'
    with open(pgmf, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster


def read_jpg(jpgf): 
    image = Image.open(jpgf)    
    return image
    
def face_window(image, left_r, left_c, size, output_shape):
    """
    Taking an image window and transform to required output_shape 
    Input: 
        image: original image
        left_r, left_c: the left up corner of crop window
        size: the required window width, aspect ratio 1:1
        output_shape: tuple (desired_h, desired_w)
    Return: a single 2d np array
    """
    
    window = image[left_r: left_r + size, left_c: left_c + size]
#    print "window size:", window.shape
    img = resize(window, output_shape, order=1, mode=None, cval=0, clip=True, preserve_range=False)
    return img

def slidding_window(image, scales, start_size, output_shape):
    """
    Taking an image and transformed into windows array
    Input: 
        image: original image
        scales: how many scales on crop window
        start_size: starting window width/height
        output_shape: tuple(desired_h, desired_w)
        output: an array of window(2d np array)
    """
    
    img = np.array(image)
    h, w = img.shape
    print 'test image size: ', img.shape, ' scales:', scales
    assert start_size*scales <= w, " start*scale out of boundary"
    assert start_size*scales <= h, " start*scale out of boundary"
    windows = []
    positions=[]
    for i in range(1,scales+1):
        size = start_size*i
        step = size/2
        
        for j in range(0,h-size+1,step):
            for k in range(0, w-size+1,step):
                s_window = face_window(img, j, k, size, output_shape)
#                print i, j, k, 'and window size', s_window.shape
#                print np.max(s_window), np.min(s_window)
                windows.append(s_window)
                positions.append((j, k, size))
    return np.array(windows), np.array(positions)

def draw_detected_window(image, positions, index):
    """
    draw windows detected as face
    Input:
        image:
        positions: list of (row, col, window_size)
        index: list of true/false, indicating which window in position to draw
    """
    fig,ax = plt.subplots(1)
    
    positions_to_draw = positions[index == 1]
    # Display the image
    ax.imshow(image)
    
    # Create a Rectangle patch
    for pos in positions_to_draw:
        size = pos[2]
        x = pos[1]
        y = pos[0]
#        print " x , y and size of windos",  x,y, size
        rect = patches.Rectangle((x+3,y+3),size-10,size,linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    plt.show()


if __name__ == '__main__':
    dataset = []
    data_to_use =1
    
    #pick a training set for face
    if data_to_use == 1:
        face_directory = 'YALE/unpadded/'
           
        for filename in os.listdir(face_directory):
            path = os.path.join(face_directory, filename)
            
            im = np.array(read_pgm(path))
            
    #       crop image and then rescale
            imx = resize(im, (49,49), order=1, mode=None, cval=0, clip=True, preserve_range=False) 
            
            fd, hog_image = hog(imx, orientations=8, pixels_per_cell=(7, 7),cells_per_block=(1, 1), transform_sqrt = True, feature_vector = True ,visualise=True)
            row = np.transpose(fd)
        #    norm = np.sqrt(np.sum(np.square(row)))      #normlization global? needed?
        #    row = row/norm
            row = np.append(row,1)
            dataset.append(row)
    else:    
        att_dir = 'att_faces'
        dataset = []
        dirs = [os.path.join(att_dir, d) for d in os.listdir(att_dir) if os.path.isdir(os.path.join(att_dir, d))]
        for dire in dirs:
            print dire
            for filename in os.listdir(dire):
                path = os.path.join(dire, filename)
    #            f = open(path, 'rb')
                
            im = np.array(read_pgm(path))
            
    #       crop image and then rescale
            imx = resize(im, (49,49), order=1, mode=None, cval=0, clip=True, preserve_range=True) 
            
            fd, hog_image = hog(imx, orientations=8, pixels_per_cell=(7, 7),cells_per_block=(1, 1), transform_sqrt = True, feature_vector = True ,visualise=True)
            row = np.transpose(fd)
        #    norm = np.sqrt(np.sum(np.square(row)))      #normlization global? needed?
        #    row = row/norm
            row = np.append(row,1)
            dataset.append(row)
        
    
    #background training data
    backg_directory = 'bground/'
    for filename2 in os.listdir(backg_directory):
        path = os.path.join(backg_directory, filename2)
        f = open(path, 'rb')
        im = np.array(read_jpg(f))
#        imx = rescale(im[0:98, 0:98], 1.0/2) # crop image and then rescale
        imx = resize(im, (49,49), order=1, mode=None, cval=0, clip=True, preserve_range=True) 
        fd, hog_image = hog(imx, orientations=8, pixels_per_cell=(7, 7),cells_per_block=(1, 1), transform_sqrt = True, feature_vector = True ,visualise=True)
        row = np.transpose(fd)

        row = np.append(row,0)
        dataset.append(row)
        
    data = np.asarray(dataset)

    X = data[:, :-1]
    y = np.transpose(data[:, -1])
    clf = svm.SVC()
    clf.fit(X, y)
#    template = clf.coef_
    
    print X.shape, y.shape, 
#    print 'max and min value of template', np.max(template), np.min(template)
    
    # testing data, predict slidding window true or false
    sample =2
    if sample == 1:
        image_path = 'simple_test.jpg'
        test_image = imread(image_path, as_grey=True)
        start_size = 130
    
    
    elif sample == 2:
        image_path = 'nasa_small.jpg'
        test_image = imread(image_path, as_grey = True)
        start_size = 55
    
    else:
        f = open('YALE/unpadded/subject02.centerlight.pgm', 'rb')
        test_image2 = read_pgm(f)
        start_size = 116
    
    windows, positions = slidding_window(test_image,1,start_size,(49,49))
    print windows.shape
    window_index = []
    features = []
    for window in windows:
        fd, hog_image = hog(window, orientations=8, pixels_per_cell=(7, 7),cells_per_block=(1, 1), transform_sqrt = True,  feature_vector = True ,visualise=True)
        features.append(fd)
        
    window_index = clf.predict(features)
    print window_index
    
    toshow_image = imread(image_path)
    draw_detected_window(toshow_image, positions, window_index)
    #templateImg = template.reshape(98,98)
    #plt.imshow(templateImg, cmap= 'gray')
    #plt.show()
    
    
    #    
    " test "   
    
    #
    #f = open('YALE/unpadded/subject02.centerlight.pgm', 'rb')
    #image = read_pgm(f)
    #arr = np.asarray(image)
    #
    #print arr.shape, arr
    #plt.imshow(arr, cmap='gray')
    #plt.show()
    
    #
    #f = open('bground/image_0573.jpg', 'rb')
    #arr = read_jpg(f)
    #
    #plt.imshow(arr, cmap= 'gray')
    #plt.show()