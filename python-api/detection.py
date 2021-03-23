import cv2
import os
import glob
import math
import numpy as np
import math
import re
from tqdm.notebook import trange, tqdm


def prepareImg(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""

    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2 # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize
    
            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
    
            kernel[i, j] = (xTerm + yTerm) * expTerm
    
    kernel = kernel / np.sum(kernel)
    return kernel

def apply_filter_threshold(img, kernelSize, sigma, theta):    
    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - imgThres
    return imgThres

def apply_dilate(img, dilate_k_size, dilate_iter):
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_k_size)
    img_dilated = cv2.dilate(img, dilate_kernel, iterations=dilate_iter)
    return img_dilated

def find_words(img):
    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, words_contours, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (words_contours, _) = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return words_contours


# Word Segmentation Function
def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0, height=3000, dilate=False, dilate_k_size=(10,5), dilate_iter=4):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
    
    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.
        height: the fixed height of the input image
        
    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # read image, prepare it by resizing it to fixed height and converting it to grayscale
    img = prepareImg(img, height=height)  # keep copy of the prepared image

    filtered_img = apply_filter_threshold(img, kernelSize, sigma, theta)

    if dilate:
        dilated_img = apply_dilate(filtered_img, dilate_k_size, dilate_iter)
    else:
        dilated_img = filtered_img
    
    words_contours = find_words(dilated_img)

    # append components to result
    results = []
    for c in words_contours:
        # skip small word candidates
        # print(cv2.contourArea(c))
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y+h, x:x+w]
        results.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    return img, sorted(results, key=lambda entry:entry[0][0])


# Right to left, top to bottom sorting algorthim for words
def sort_words(res):
    boxes = []
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        boxes.append((x,y,w,h,wordImg))
    # sort all rect by their y
    boxes.sort(key=lambda b: b[1])
    # initially the line bottom is set to be the bottom of the first rect
    line_begin_indices = []
    line_bottom = boxes[0][1]+boxes[0][3]-1
    line_begin_idx = 0
    for i in range(len(boxes)):
        # when a new box's top is below current line's bottom
        # it's a new line
        if boxes[i][1] > line_bottom:
            # sort the previous line by their x
            boxes[line_begin_idx:i] = sorted(boxes[line_begin_idx:i], key=lambda b: b[0], reverse=True)
            line_begin_idx = i
            line_begin_indices.append(i)
        # regardless if it's a new line or not
        # always update the line bottom
        line_bottom = max(boxes[i][1]+boxes[i][3]-1, line_bottom)
    # sort the last line
    boxes[line_begin_idx:] = sorted(boxes[line_begin_idx:], key=lambda b: b[0], reverse=True)
    return boxes, line_begin_indices

def main(img):
    # kernelSize=25, sigma=11, theta=7, minArea=0, height=3000, dilate=False, dilate_k_size=(10,5), dilate_iter=4
    _, res = wordSegmentation(img, kernelSize=25, sigma=11, theta=11, minArea=1000, height=2000 , dilate=True, dilate_k_size=(8,10), dilate_iter=12) # segment the image into words
    # the output of wordSegmentation is list of tuples
    sorted_words, line_begin_indices = sort_words(res)
    words = []
    for (i, w) in tqdm(enumerate(sorted_words), desc="Saving Words", leave=False):
        (_, _, _, _, word_img) = w # the tuple is composed of bounding box coords and word image we only care about the image in this code
        words.append(word_img)
    return words, line_begin_indices