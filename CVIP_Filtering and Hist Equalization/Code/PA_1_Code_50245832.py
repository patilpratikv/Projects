
# coding: utf-8

# In[5]:


from skimage import color
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

#Fundtion def for convolution either direct 2D or 2D Separable
def convolve(image, filter_size, kernel_filter_1, kernel_filter_2 = None):
    
    # convolution output
    output = np.zeros_like(image)            
    # Add zero padding to the input image
    if filter_size % 2 == 1:
        padd_size = int((filter_size-1)/2)
        image_padded = np.zeros((image.shape[0] + 2 * padd_size, image.shape[1] + 2 * padd_size))
        image_padded[padd_size:-padd_size, padd_size:-padd_size] = image
        # Loop over every pixel of the image
        for x in range(image.shape[1]):     
            for y in range(image.shape[0]):
                # element-wise multiplication of the filter and the image
                if kernel_filter_2 is None:
                    output[y,x] = (kernel_filter_1*image_padded[y:y+filter_size,x:x+filter_size]).sum() 
                else:
                    output[y,x] = (np.multiply(kernel_filter_1, np.sum(np.multiply(kernel_filter_2,image_padded[y:y+filter_size,x:x+filter_size]), axis=1))).sum()
    else:
        print('Filter should be ODD SQUARE matrix!!!')
                    
    return output

def question_one():
    # 2D Filter declaration 3*3
    twoD_vert_sobel_filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    twoD_horz_sobel_filter = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    # 2D Separable filters 3*3
    oneD_kernel_sobel_1 = np.array([[-1,0,1]])
    oneD_kernel_sobel_2 = np.array([[1,2,1]])
    # 2D Separable filters 100*100
    oneD_kernel_1 = np.ones((1, 101))
    oneD_kernel_2 = np.ones((101, 1))
    twoD_kernel = np.outer(oneD_kernel_1, oneD_kernel_2)

    #Reading image as gray
    img = io.imread('lena_gray.jpg', as_grey=True)

    # Direct 2-D edge detection
    #--------------------------------------------------------------------------
    twoD_Horizontal_Edge_detect = convolve(img, 3, twoD_horz_sobel_filter)
    twoD_Vertical_Edge_detect   = convolve(img, 3, twoD_vert_sobel_filter)
    twoD_Absolute_Edge          = np.sqrt(np.add(np.power(twoD_Horizontal_Edge_detect, 2), np.power(twoD_Vertical_Edge_detect, 2)))
    #Plotting of all Figures
    print("Images with direct 2D Convolution\n")
    plt.figure(1)
    plt.imshow(twoD_Horizontal_Edge_detect, cmap='gray')
    plt.title('Horizontal Edge Detect')
    plt.figure(2)
    plt.imshow(twoD_Vertical_Edge_detect, cmap='gray')
    plt.title('Vertical Edge Detect')
    plt.figure(3)
    plt.imshow(twoD_Absolute_Edge, cmap='gray')
    plt.title('Absolute Edge Detect')
    plt.show()

    # 1-D  separable edge detection
    #--------------------------------------------------------------------------
    oneD_Horizontal_Edge_detect = convolve(img, 3, oneD_kernel_sobel_1, oneD_kernel_sobel_2)
    oneD_Vertical_Edge_detect   = convolve(img, 3, oneD_kernel_sobel_2, oneD_kernel_sobel_1)
    oneD_Absolute_Edge          = np.sqrt(np.add(np.power(oneD_Horizontal_Edge_detect, 2), np.power(oneD_Vertical_Edge_detect, 2)))
    #Plotting of all Figures
    print("Images with direct 1D Separable filter Convolution\n")
    plt.figure(4)
    plt.imshow(oneD_Horizontal_Edge_detect, cmap='gray')
    plt.title('Horizontal Edge Detect')
    plt.figure(5)
    plt.imshow(oneD_Vertical_Edge_detect, cmap='gray')
    plt.title('Vertical Edge Detect')
    plt.figure(6)
    plt.imshow(oneD_Absolute_Edge, cmap='gray')
    plt.title('Absolute Edge Detect')
    plt.show()

    # Time comparison for direct 2D and 1D Separable convolution with 100*100 averaging filter ie. all ones.
    start = time.clock()
    oneD_Averaging_filter = convolve(img, 101, oneD_kernel_1, oneD_kernel_2)
    print("Time required for 2D Separable filters : %f seconds" % (time.clock() - start))
    start = time.clock()
    twoD_Averaging_filter = convolve(img, 101, twoD_kernel)
    print("Time required for direct 2D filters : %f seconds" % (time.clock() - start))
    
def question_two():
    # Image gray levels
    gray_levels = 256
    # Reading Color Image
    clr_img = matplotlib.pyplot.imread('pout.JPG')
    # Converting to gray
    gray_img = color.rgb2gray(clr_img)  
    # Scaling between [0-255]
    scaled_gray_img = ((gray_levels)*gray_img).astype('uint8') 
    # Array to store intensity count
    intensity_count = np.zeros(gray_levels).astype('uint16') 
    # Intensity value count for each level
    scaled_list = scaled_gray_img.ravel().tolist()
    for i in range(gray_levels):
        intensity_count[i] = scaled_list.count(i)

    # Cumulutive intensity Calculation
    cum_intensity_count = np.cumsum(intensity_count)
    # Scaling factor
    scaling_factor = (gray_levels - 1)/(scaled_gray_img.shape[0] * scaled_gray_img.shape[1]) 
    # Calculate new intensity values
    new_intensity_level = scaling_factor * cum_intensity_count
    # Create buffer for new image    
    new_image = np.zeros_like(scaled_gray_img) 
    # Copying new intensity values to new image matrix
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i,j] = new_intensity_level[scaled_gray_img[i,j]]

    # Histogram for old image
    plt.figure(1)
    plt.bar(list(range(0,256)),intensity_count)
    plt.title('Original Image Histogram')
    plt.xlabel('Intensity Level')
    plt.ylabel('# of Pixels')
    # Cumulitive Histogram for old image
    plt.figure(2)
    plt.bar(list(range(0,256)),cum_intensity_count)
    plt.title('Original Image Cumulative Histogram')
    plt.xlabel('Intensity Level')
    plt.ylabel('# of Pixels')
    # Transformation function
    plt.figure(3)
    plt.plot(new_intensity_level)
    plt.title('Transformation Function')
    plt.xlabel('Old Intensity Level')
    plt.ylabel('New Intensity Level')
    # Histogram for new image
    plt.figure(4)
    plt.hist(new_image.ravel(),256,[0,256])
    plt.title('Contrast adjusted Histogram')
    plt.xlabel('Intensity Level')
    plt.ylabel('# of Pixels')
    # Old Image Display
    plt.figure(5)
    plt.imshow(scaled_gray_img, cmap='gray')
    plt.title('Old Dark Image')
    # New Image Display
    plt.figure(6)
    plt.imshow(new_image, cmap='gray')
    plt.title('New Well Lit Image')
    plt.show()
    
print("Programming Assignment 1\n")
print("Executing solution for question 1\n")
question_one()
print("Question 1 solved\n")
print("Executing solution for question 2\n")
question_two()
print("Question 2 solved\n")
print("Assignment Done!!!")

