
# coding: utf-8

# In[ ]:


from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from numpy import nan
import numpy as np
import matplotlib
import random
import time
import cv2


def doPadding(image, filter_size):
    # Add padding to image with respective filter size
    if filter_size % 2 == 1:
        padd_size = int((filter_size-1)/2)
        image_padded = np.zeros((image.shape[0] + 2 * padd_size, image.shape[1] + 2 * padd_size))
        image_padded[padd_size:-padd_size, padd_size:-padd_size] = image
        
        return image_padded
    else:
        print('Box size should be ODD SQUARE matrix!!!')

def disparity(firstImage, secondImage, boxSize, dispMap, dispSide):
    # height and width of image
    height, width = firstImage.shape
    # Pad left and right images
    padded_first_gray_img = doPadding(firstImage, boxSize)
    padded_second_gray_img = doPadding(secondImage, boxSize)
    # Array to store disparity values
    disparityArray = np.zeros_like(firstImage)
    # Look at 70 pixel to left and right, from current column to find the similar point
    disparityRange = 70
    # for each row
    for row in range(height):
        # for each column
        for col in range(width):
            # Selecting matrix from first image to compare
            template = padded_first_gray_img[row:row + boxSize, col:col + boxSize]
            if dispSide == 'left':
                # Left end of search array in right image
                leftSearchEnd = max(0, col - disparityRange)
                # Right end of search array in right image
                rightSearchEnd = col + 1
            elif dispSide == 'right':
                # Left end of search array in right image
                leftSearchEnd = col
                # Right end of search array in right image
                rightSearchEnd = min(width, col + disparityRange)
            # Array to store ssd values for each pixel
            ssdArray = np.zeros((rightSearchEnd - leftSearchEnd)) 
            index = 0
            # Traverse through range defined above in second image
            for col_2 in range(leftSearchEnd, rightSearchEnd):
                # Selecting matrix from second image to find ssd
                blockToMatch = padded_second_gray_img[row:row + boxSize, col_2:col_2 + boxSize]
                ssdArray[index] = np.sum(np.power(blockToMatch - template, 2))
                index += 1
            # Stoaring disparity value in the array
            if dispSide == 'left':
                disparityArray[row, col] = col - (np.argsort(ssdArray)[0] + leftSearchEnd)
            elif dispSide == 'right':
                disparityArray[row, col] = np.argsort(ssdArray)[0]
    
    return disparityArray, np.mean(np.square(disparityArray - dispMap))

def consistency_check(firstImage, secondImage, groundTruthDisp, side = None):
    '''This function computer consistency matrix and calculates MSE with ground truth image too.
    eg. consistencyMatrix, MSE = consistency_check(firstImage, secondImage, left/right, groundTruth)
    input argument:
        firstImage - Disparity matrix of image whos consistency needs to find (Grayscale matrix of Image)
        secondIMage - Disparity matrix of image with which consistency needs to find (Grayscale matrix of Image)
        groundTruth - left or right groundtruth image as per the 4th argument (Grayscale matrix of image)
        left/right - left or right consistency (String Input)
    output argumet:
        consistencyMatrix - left or right consistency matrix of image as per the 4th input argument
        MSE - MSE of computed consistencyMatrix and groundTruth image
    Sample use of function:
        leftConsistencyMat, leftMSE = consistency_check(left_gray_image, right_gray_image, 'left', left_disparith_groundtruth)'''
    
    consistencyMat = np.zeros_like(firstImage)
    height, width = firstImage.shape
    trueVal = []
    calcVal = []
    for row in range(height):
        for col in range(width):
            pixelCompareWith = firstImage[row, col]
            if side == None or side == 'left':
                colInOtherImage = max(0, col - pixelCompareWith)
            else:
                colInOtherImage = min(width, col + pixelCompareWith)
            pixelCompareTo = secondImage[row, colInOtherImage]
            
            if pixelCompareWith == pixelCompareTo:
                consistencyMat[row, col] = pixelCompareWith
                trueVal.append(groundTruthDisp[row, col])
                calcVal.append(pixelCompareWith)
            else:
                consistencyMat[row, col] = 0
                
    return consistencyMat, np.mean(np.square(np.asarray(trueVal[:]) - np.asarray(calcVal[:])))

def doSynthesis(left_clr_img, right_clr_img, left_disp_gray_img, right_disp_gray_img):
    # Array to store synthesis values
    view_synthesis = np.zeros_like(left_clr_img).astype('uint8')
    # for each row
    for row in range(left_disp_gray_img.shape[0]):
        # for each column
        for col in range (left_disp_gray_img.shape[1]):
            # Midway point of both disparity values
            mid_index = int(left_disp_gray_img[row, col]/2)
            # If valid value
            if col - mid_index >= 0:
                view_synthesis[row, col-mid_index, :] = left_clr_img[row, col, :]
    
    for row in range(right_disp_gray_img.shape[0]):
        for col in np.arange(right_disp_gray_img.shape[1]-1, 0, -1):
            mid_index = int(right_disp_gray_img[row, col]/2)
            if (col + mid_index < right_disp_gray_img.shape[1]):            
                view_synthesis[row, col + mid_index, :] = right_clr_img[row, col, :]
                
    # https://stackoverflow.com/questions/40067243/matplotlib-adding-blue-shade-to-an-image
    view_synthesis = cv2.cvtColor(view_synthesis, cv2.COLOR_BGR2RGB)
    # Plotting synthesized value
    plt.figure(figsize = (9,6))
    plt.imshow(view_synthesis, aspect='auto')
    plt.title('Synthesized View')


def disparityAndConsistencyCheck(left_gray_img, right_gray_img,  left_disp_gray_img, right_disp_gray_img):
    # box size = 3
    boxSize = 3
    # Disparity estimate of left and right image with 3x3 Filter
    # ---------------------------------------------------------------------------
    left_disparityArray_3x3, leftMSE_3x3 = disparity(left_gray_img, right_gray_img, boxSize, left_disp_gray_img, 'left')
    plt.figure(1)
    plt.imshow(left_disparityArray_3x3, cmap='gray')
    plt.title('Left Disparity with 3x3 Filter')
    plt.show()
    print('Left disparity MSE with 3x3 Filter = {0}.'.format(leftMSE_3x3))
    right_disparityArray_3x3, rightMSE_3x3 = disparity(right_gray_img, left_gray_img, boxSize, right_disp_gray_img, 'right')
    plt.figure(2)
    plt.imshow(right_disparityArray_3x3, cmap='gray')
    plt.title('Right Disparity with 3x3 Filter')
    plt.show()
    print('Right disparity MSE with 3x3 Filter = {0}.'.format(rightMSE_3x3))

    # Check consistency of Left and Right disparity maps with 3x3 Filter
    # ---------------------------------------------------------------------------
    leftConsistencyMat_3x3, leftMSE_3x3_aftrConsistency = consistency_check(left_disparityArray_3x3, right_disparityArray_3x3, left_disp_gray_img, 'left')
    plt.figure(3)
    plt.imshow(leftConsistencyMat_3x3, cmap='gray')
    plt.title('Left Consistency with 3x3 Filter')
    plt.show()
    print('Left disparity MSE after consistency check with 3x3 Filter = {0}.'.format(leftMSE_3x3_aftrConsistency))
    rightConsistencyMat_3x3, rightMSE_3x3_aftrConsistency = consistency_check(right_disparityArray_3x3, left_disparityArray_3x3, right_disp_gray_img, 'right')
    plt.figure(4)
    plt.imshow(rightConsistencyMat_3x3, cmap='gray')
    plt.title('Right Consistency with 3x3 Filter')
    plt.show()
    print('Right disparity MSE after consistency check with 3x3 Filter = {0}.'.format(rightMSE_3x3_aftrConsistency))

    # box size = 9
    boxSize = 9
    # Disparity estimate of left and right image with 9x9 Filter
    # ---------------------------------------------------------------------------
    left_disparityArray_9x9, leftMSE_9x9 = disparity(left_gray_img, right_gray_img, boxSize, left_disp_gray_img, 'left')
    plt.figure(5)
    plt.imshow(left_disparityArray_9x9, cmap='gray')
    plt.title('Left Disparity with 9x9 Filter')
    plt.show()
    print('Left disparity MSE with 9x9 with 9x9 Filter = {0}.'.format(leftMSE_9x9))
    right_disparityArray_9x9, rightMSE_9x9 = disparity(right_gray_img, left_gray_img, boxSize, right_disp_gray_img, 'right')
    plt.figure(6)
    plt.imshow(right_disparityArray_9x9, cmap='gray')
    plt.title('Right Disparity with 9x9 Filter')
    plt.show()
    print('Right disparity MSE with 9x9 with 9x9 Filter = {0}.'.format(rightMSE_9x9))

    # Check consistency of Left and Right disparity maps with 9x9 Filter
    # ---------------------------------------------------------------------------
    leftConsistencyMat_9x9, leftMSE_9x9_aftrConsistency = consistency_check(left_disparityArray_9x9, right_disparityArray_9x9, left_disp_gray_img, 'left')
    plt.figure(7)
    plt.imshow(leftConsistencyMat_9x9, cmap='gray')
    plt.title('Left Consistency with 9x9 Filter')
    plt.show()
    print('Left disparity MSE with 9x9 after consistency check with 9x9 Filter = {0}.'.format(leftMSE_9x9_aftrConsistency))
    rightConsistencyMat_9x9, rightMSE_9x9_aftrConsistency = consistency_check(right_disparityArray_9x9, left_disparityArray_9x9, right_disp_gray_img, 'right')
    plt.figure(8)
    plt.imshow(rightConsistencyMat_9x9, cmap='gray')
    plt.title('Right Consistency with 9x9 Filter')
    plt.show()
    print('Right disparity MSE with 9x9 after consistency check with 9x9 Filter = {0}.'.format(rightMSE_9x9_aftrConsistency))

def segmentImage(imagePath, h, tolerance):
    '''This function will perform the image segmentation using meanshift algorithm
    Input Arguments:
        imagePath: Path of the image which needs to be segmented
        h: blob size used to find the mean
        tolerance: Shift in previous and current mean
    Use of Function:
    segmentImage('Butterfly.jpg', 140, 20)'''
    
    # Reading Image
    Butterfly_img = mpimg.imread(imagePath)
    
    start = time.time()
    # Stoaring dimension of image
    height, width, _ = Butterfly_img.shape

    # Declaration and initialization of 5 dimensional feature array
    featureMatrix = np.zeros((height * width, 5), dtype='float')
    # Declaration and initialization of new segmented image
    segmentImage = np.zeros_like(Butterfly_img)
    
    # Hyper parameter declaration
    # -------------------------------------
    # Use if not calling from function
    # Size of blob
    # h = 140
    # Shift in mean
    # tolerance = 20
    # -------------------------------------
    
    # Index to access FeatureMatrix elements
    featureIndex = 0
    # Store image pixel values and positions in featurematrix
    # For each row
    for row in range(height):
        # For each column
        for col in range(width):
            # Store x position
            featureMatrix[featureIndex, 3] = row
            # Store y position
            featureMatrix[featureIndex, 4] = col
            # Store R, G, B value of pixel seperatly from RGB (Vectorized) 
            featureMatrix[featureIndex, 0:3] = Butterfly_img[row, col, 0:3]
            featureIndex += 1

    # Reinitialize featureindex to access featurematrix elements
    featureIndex = 1
    # These are special variable declared if previous meanshift and current meanshift 
    # shifted by very small amount (but not 0)for 10 succesive loops, we choose new random row in featurematrix
    prev_meanShift = 0
    count = 0
    small_shift = 0.000001

    # Keep on assignning element till FeatureMatrix is empty
    while featureMatrix.shape[0] >= 1:
        # Choose new random element for fresh start only if we have reached the dense area otherwise use last mean for iteration
        if featureIndex == 1:
            # Select randomrow from featureMatrix
            index = random.randint(0, featureMatrix.shape[0] - 1)
            # Select row elements as current means
            curr_mean = featureMatrix[index,:].reshape(-1, 5)
            featureIndex = 0
            count = 0
        # Find ecludian distance from current mean to all other elemnts in featureMatrix (Vectorized)
        eucldn_dist = euclidean_distances(curr_mean, featureMatrix)
        # Sort all ecludian distances by value in ascending order 
        dist_sortAscend = np.sort(eucldn_dist)
        # Sort all ecludian distances by index in ascending order 
        dist_sortAscend_index = np.argsort(eucldn_dist)
        # Choose all elemnts whose distance is less than size of blob (h)
        _, interestElements = np.where(dist_sortAscend < h)
        if interestElements.any() == 0:
            # If there are no elemnts in blob region assign current mean as new mean
            new_mean = curr_mean
        else:
            # else calculate new mean for elements within blob(h) (Vectorized)
            new_mean = np.mean(featureMatrix[dist_sortAscend_index[0, 0:interestElements[-1]]], axis = 0).reshape(-1,5)
        # Calculate mean shift for previous and current mean (Vectorized)
        meanShift = euclidean_distances(curr_mean, new_mean)
        # This loop is for special rare scenarios if current and previous mean shift is very small but not zero
        if abs(meanShift - prev_meanShift) < small_shift:
            count += 1
            if count == 10:
                featureIndex = 1
        # Assign current meanShift to variable
        prev_meanShift = meanShift

        if meanShift == 0:
            # If the meanshift is zero assign current means to selected pixels within blob (Vectorized)
            segmentImage[featureMatrix[dist_sortAscend_index[0,0], 3].astype('uint16'), featureMatrix[dist_sortAscend_index[0,0], 4].astype('uint16'), 0:3] = curr_mean[0, 0:3]
            # Delete all the pixels from featurematrix after assigning to segemented matrix above
            featureMatrix = np.delete(featureMatrix, dist_sortAscend_index[0,0].reshape(-1, 1), axis = 0)
            # Set variable to one to select new random row in updated feature matrix
            featureIndex = 1
        elif meanShift < tolerance:
            # If the meanshift is less than tolerance value then assign new means to selected pixels within blob (Vectorized)
            segmentImage[featureMatrix[dist_sortAscend_index[0, 0:interestElements[-1]], 3].astype('uint16'), featureMatrix[dist_sortAscend_index[0, 0:interestElements[-1]], 4].astype('uint16'), 0:3] = new_mean[0, 0:3]
            # Delete all the pixels from featurematrix after assigning to segemented matrix above
            featureMatrix = np.delete(featureMatrix, dist_sortAscend_index[0, 0:interestElements[-1]].reshape(-1, 1), axis = 0)
            # Set variable to one to select new random row in updated feature matrix
            featureIndex = 1
        else:
            # If meanshift is more than tolerance values then assign new means to current means and iterate the process
            curr_mean = new_mean
    print("Time takend to complete Segmentation is {0}seconds.".format(time.time()-start))
    # Display segmented image
    plt.figure(figsize=(10,5))
    plt.imshow(segmentImage)
    plt.title("Segmented Image")
    plt.show()

def dymanicProgramming(left_gray_img, right_gray_img):
    numcols = left_gray_img.shape[1]
    numrows = left_gray_img.shape[0]
    disparity = np.zeros_like(left_gray_img)
    #Disparity Computation for Left Image

    OcclusionCost = 10 #(You can adjust this, depending on how much threshold you want to give for noise)

    #For Dynamic Programming you have build a cost matrix. Its dimension will be numcols x numcols

    CostMatrix = np.zeros((numcols,numcols))
    DirectionMatrix = np.zeros((numcols,numcols))  #(This is important in Dynamic Programming. You need to know which direction you need traverse)

    #We first populate the first row and column values of Cost Matrix
    for row in range(0,numrows):
        for k in range(0, numcols):
            CostMatrix[k,0] = k*OcclusionCost
            CostMatrix[0,k] = k*OcclusionCost
        for i in range(0, numcols):    
            for j in range(0, numcols):
                min1 = CostMatrix[i-1, j-1] + (left_gray_img[row,i]-right_gray_img[row,j])
                min2 = CostMatrix[i-1,j] + OcclusionCost
                min3 = CostMatrix[i,j-1] + OcclusionCost
                CostMatrix[i,j] = cmin = min(min1, min2, min3)

                if min1 == cmin:
                    DirectionMatrix[i,j] = 1
                if min2 == cmin:
                    DirectionMatrix[i,j] = 2
                if min3 == cmin:
                    DirectionMatrix[i,j] = 3

        i = j = numcols
        while i != 0 and j!=0:
            if DirectionMatrix[i-1,j-1] == 1:
                disparity[row, i-1] = i-j
                i -= 1
                j -= 1
            elif DirectionMatrix[i-1,j-1] == 2:
                i -= 1
            elif DirectionMatrix[i-1,j-1] == 3:
                j -= 1

    plt.figure()
    plt.imshow(disparity, cmap='gray')
    plt.title('Disparity map by dynamic programming')
    
# Main Program starts here
# ----------------------------------------------------------------------------
# Reading images
left_clr_img = cv2.imread('view1.png')
right_clr_img = cv2.imread('view5.png')
left_gray_img = cv2.imread('view1.png', 0)
right_gray_img = cv2.imread('view5.png', 0)
# Reading true disparities 
left_disp_gray_img = cv2.imread('disp1.png', 0)
right_disp_gray_img = cv2.imread('disp5.png', 0)

# Disparity and Consistency Check
# ----------------------------------------------------------------------------
disparityAndConsistencyCheck(left_gray_img, right_gray_img,  left_disp_gray_img, right_disp_gray_img)

# Dynamic Programming for Disparity
# ----------------------------------------------------------------------------
dymanicProgramming(left_gray_img, right_gray_img)

# View Sysnthesis
# ----------------------------------------------------------------------------
doSynthesis(left_clr_img, right_clr_img, left_disp_gray_img, right_disp_gray_img)

# Image Segmentation
# ----------------------------------------------------------------------------
segmentImage('Butterfly.jpg', 140, 20)
