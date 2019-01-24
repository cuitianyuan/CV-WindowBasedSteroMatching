import cv2
import numpy as np

def disparity_ssd(img1, img2, direction, w_size, dmax):
    """Returns a disparity map D(y, x) using the Sum of Squared Differences.

    Assuming img1 and img2 are the left (L) and right (R) images from the same scene. The disparity image contains
    values such that: L(y, x) = R(y, x) + D(y, x) when matching from left (L) to right (R).

    This method uses the Sum of Squared Differences as an error metric. Refer to:
    https://software.intel.com/en-us/node/504333

    The algorithm used in this method follows the pseudocode:

    height: number of rows in img1 or img2.
    width: number of columns in img1 or img2.
    DSI: initial array containing only zeros of shape (height, width, dmax)
    kernel: array of shape (w_size[0], w_size[1]) where each value equals to 1/(w_size[0] * w_size[1]). This allows
            a uniform distribution that sums to 1.

    for d going from 0 to dmax:
        shift = some_image_shift_function(img2, d)
        diff = img1 - shift  # SSD
        Square every element values  # SSD
        Run a 2D correlation filter (i.e. cv.filter2D) using the kernel defined above
        Save the results in DSI(:, :, d)

    For each location r, c the SSD for an offset d is in DSI(r,c,d). The best match for pixel r,c is represented by
    the index d for which DSI(r,c,d) is smallest.

    Args:
        img1 (numpy.array): grayscale image, in range [0.0, 1.0].
        img2 (numpy.array): grayscale image, in range [0.0, 1.0] same shape as img1.
        direction (int): if 1: match right to left (shift img1 left).
                         if 0: match left to right (shift img2 right).
        w_size (tuple): window size, type int representing both height and width (h, w).
        dmax (int): maximum value of pixel disparity to test.

    Returns:
        numpy.array: Disparity map of type int64, 2-D array of the same shape as img1 or img2.
                     This array contains the d values representing how far a certain pixel has been displaced.
                     Return without normalizing or clipping.
    """

    w, h = img1.shape 
    ssd3 = np.zeros((w,h,dmax))
    sq_diff = np.zeros((w,h,dmax))
    dsi = np.zeros((w,h,dmax))
    depth = np.zeros((w,h))
        # np.roll to control burden

    for d in range(dmax): 
        rshift = np.roll(img2,d)
        sq_diff[:, :, d] = (img1 - rshift) **2  #    square every element values  # SSD
    if direction == 0 :
        for y in range(0, h - w_size[0]): 
            for  x in range(0, w - w_size[1]):
                sq_diff_kernel = sq_diff[ x:(x+w_size[0] ), y:(y+w_size[1]) , : ]
                sum_diff_sq_kernel = np.sum(np.sum(sq_diff_kernel, axis=0), axis=0)
                d = sum_diff_sq_kernel.argmin()
                depth[x, y] = d
    else:
        for y in range(0, h - w_size[0]):
            for  x in reversed(range( w_size[1], w )):
#                print 'y=', y, 'x=', x,'from',(x-w_size[0]+1),'to',x
                sq_diff_kernel = sq_diff[ (x-w_size[0]+1):(x+1), y:(y+w_size[1]), : ]
                sum_diff_sq_kernel = np.sum(np.sum(sq_diff_kernel, axis=0), axis=0)
                d = sum_diff_sq_kernel.argmin()
                depth[x, y] = d
                    
    return depth.astype(int)   
 
 
def disparity_ncorr(img1, img2, direction, w_size, dmax):
    """Returns a disparity map D(y, x) using the normalized correlation method.

    This method uses a similar approach used in disparity_ssd replacing SDD with the normalized correlation metric.

    For more information refer to:
    https://software.intel.com/en-us/node/504333

    Unlike SSD, the best match for pixel r,c is represented by the index d for which DSI(r,c,d) is highest.

    Args:
        img1 (numpy.array): grayscale image, in range [0.0, 1.0].
        img2 (numpy.array): grayscale image, in range [0.0, 1.0] same shape as img1.
        direction (int): if 1: match right to left (shift img1 left).
                         if 0: match left to right (shift img2 right).
        w_size (tuple): window size, type int representing both height and width (h, w).
        dmax (int): maximum value of pixel disparity to test.

    Returns:
        numpy.array: Disparity map of type int64, 2-D array of the same shape size as img1 or img2.
                     This array contains the d values representing how far a certain pixel has been displaced.
                     Return without normalizing or clipping.
    """

    w, h = img1.shape 
    depth = np.zeros((w,h)) 
     
    left_img = img1.astype(np.float32)
    right_img = img2.astype(np.float32)
    
    if direction == 0 : 
        for y in range(dmax, h - w_size[0]): #for y in range(0, h - w_size[0]): #smooth
            for  x in range(0, w - w_size[1]):            
                kernel_l = left_img[x:(x+w_size[0] ), y:(y+w_size[1])]
#                kernel_r = right_img[x:(x+w_size[0]), max((y-dmax),0):(y+w_size[1])]  # Smooth edge
                kernel_r = right_img[x:(x+w_size[0]), (y-dmax) :(y+w_size[1])] 
#                print kernel_l.shape, kernel_r.shape
                sim = cv2.matchTemplate(kernel_l, kernel_r  ,  cv2.TM_CCOEFF_NORMED)  
#                sim = cv2.matchTemplate(kernel_l, kernel_r  ,  cv2.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sim)
#                print min_val, max_val, min_loc, max_loc
                depth[x, y] = dmax - max_loc[0] 
                               
    if direction == 1 : 
        for y in reversed(range(dmax, h - w_size[0])): #range(dmax, h - w_size[0]):
            for  x in reversed(range( w_size[1], w )):
                kernel_l = left_img[(x-w_size[0]+1):(x+1), y:(y+w_size[1])]
                kernel_r = right_img[(x-w_size[0]+1):(x+1), (y-dmax) :(y+w_size[1])]    
                sim = cv2.matchTemplate(kernel_l, kernel_r  ,  cv2.TM_CCOEFF_NORMED)   
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sim)
                depth[x, y] = dmax - max_loc[0] 
                                                                          
    return depth.astype(int)  
         
def add_noise(img, sigma):
    """Returns a copy of the input image with gaussian noise added. The Gaussian noise mean must be zero.
    The parameter sigma controls the standard deviation of the noise.

    Args:
        img (numpy.array): input image of type int or float.
        sigma (float): gaussian noise standard deviation.

    Returns:
        numpy.array: output image with added noise of type float64. Return it without normalizing or clipping it.
    """
    r, c = img.shape
    im = np.zeros((r,c), np.float64)
    cv2.randn(im, (0) , (sigma) )
    return (img.astype(np.float64) + im)
    


def increase_contrast(img, percent):
    """Returns a copy of the input image with an added contrast by a percentage factor.

    Args:
        img (numpy.array): input image of type int or float.
        percent (int or float): value to increase contrast. The autograder uses percentage values i.e. 10%.

    Returns:
        numpy.array: output image with added noise of type float64. Return it without normalizing or clipping it.
    """

    img1 = np.copy(img)
    img1 = img.astype(np.float64)
    img1 = cv2.multiply(img1, np.array([1.0 + float(percent)/100]))  
    return img1
