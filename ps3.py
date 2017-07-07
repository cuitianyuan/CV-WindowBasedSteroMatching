import cv2
import numpy as np
#from matplotlib import pyplot as plt
#plt.imshow(depth,'gray')


def disparity_ssd(img1, img2, direction, w_size, dmax):
#    img1 = l
#    img2 = r
#    direction = 0
#    w_size = (5,5)
#    dmax = 3
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
#
#print depth[50,50]
#print np.min(depth)
#print np.max(depth)
# 
#print np.sum(depth)
#
#
#
### TEst result
##rshift = np.roll(img2,2)
##img1[50:52,50:52]
##rshift[50:52,50:52]
##diff = img1 - rshift
##cv2.imwrite(os.path.join(output_dir, 'test.png'), normalize_and_scale(diff))
#    #    Run a 2D correlation filter and the kernel defined above #  cv.filter2D(...)  
#
#
#
##AssertionError: Disparity map is incorrect. Input images are similar to pair0.
##The output disparity map does not have a centered square with the expected size. 
##Expected size: 10100. Student size: 17252. 
##Max difference allowed: 1010. 
##Testing parameters - w_size: (5, 5). dmax: 3. 
##direction: 0
#
#
#
#def stereo_match(left_img, right_img, kernel, max_offset):
#    # Load in both images, assumed to be RGBA 8bit per channel images
#    left_img = l
#    right_img = r
#    kernel = 4
#    max_offset = 3
#    
#    w, h = left_img.shape  # assume that both images are same size   
#    
#    # Depth (or disparity) map
#    depth = np.zeros((w, h), np.uint8)
#    depth.shape = h, w
#       
#    kernel_half = int(kernel / 2)    
##    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
#      
#    for y in range(kernel_half, h - kernel_half):      
##        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)
#        
#        for x in range(kernel_half, w - kernel_half):
#            best_offset = 0
#            prev_ssd = 65534
#            
#            for offset in range(max_offset):               
#                ssd = 0
#                ssd_temp = 0                            
#                
#                # v and u are the x,y of our local window search, used to ensure a good 
#                # match- going by the squared differences of two pixels alone is insufficient, 
#                # we want to go by the squared differences of the neighbouring pixels too
#                for v in range(-kernel_half, kernel_half):
#                    for u in range(-kernel_half, kernel_half):
#                        # iteratively sum the sum of squared differences value for this block
#                        # left[] and right[] are arrays of uint8, so converting them to int saves
#                        # potential overflow, and executes a lot faster 
##                        print off_set, u, v,  left_img[y+v, x+u] ,  right_img[y+v, (x+u) - offset], left_img[y+v, x+u]  -  right_img[y+v, (x+u) - offset] 
#                        ssd_temp =  left_img[y+v, x+u]  -  right_img[y+v, (x+u) - offset] 
#                        ssd += ssd_temp * ssd_temp              
#                
#                # if this value is smaller than the previous ssd at this block
#                # then it's theoretically a closer match. Store this value against
#                # this block..
#                if ssd < prev_ssd:
#                    prev_ssd = ssd
#                    best_offset = offset
#                            
#            # set depth output for this x,y location to the best match
#            depth[y, x] = best_offset 
#                                
#    # Convert to PIL and save it
#    return depth
#
#print depth[50,50]
#print np.min(depth)
#print np.max(depth)
# 
#np.sum(depth)
# 7805
# 
 
 
 
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
     
     
#    img1 = cv2.imread(os.path.join(input_dir, 'pair0-L.png'), 0)  / 255.
#    img2 = cv2.imread(os.path.join(input_dir, 'pair0-R.png'), 0)  / 255.
#    img1 = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
#    img2 = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0)  / 255.
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
#                 print kernel_l.shape, kernel_r.shape
                sim = cv2.matchTemplate(kernel_l, kernel_r  ,  cv2.TM_CCOEFF_NORMED)   
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sim)
                depth[x, y] = dmax - max_loc[0] 
#                print min_val, max_val, min_loc, max_lo
#
#    if direction == 1 : 
#        for y in reversed(range(0, h - w_size[0]-dmax)): #range(dmax, h - w_size[0]):
#            for  x in reversed(range( w_size[1], w )):
#                kernel_l = left_img[(x-w_size[0]+1):(x+1), y:(y+w_size[1])]
#                kernel_r = right_img[(x-w_size[0]+1):(x+1), (y) :(y+w_size[1]+dmax)]    
##                 print kernel_l.shape, kernel_r.shape
#                sim = cv2.matchTemplate(kernel_l, kernel_r ,  cv2.TM_CCOEFF_NORMED)   
#                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sim)
##                print min_val, max_val, min_loc, max_loc
#                depth[x, y] =  max_loc[0]
                                                                          
    return depth.astype(int)  
                
#        print depth[0,0]
#        print depth[20,20]
#        print depth[50,50]
#        print depth[70,70]
#        print depth[100,100]
#        plt.imshow(depth,'gray')
#        depth1 = depth
#        plt.imshow(normalize_and_scale(depth ),'gray')
#        plt.imshow(normalize_and_scale(depth1),'gray')
#        plt.imshow(normalize_and_scale(depth0),'gray')
         
def add_noise(img, sigma):
    """Returns a copy of the input image with gaussian noise added. The Gaussian noise mean must be zero.
    The parameter sigma controls the standard deviation of the noise.

    Args:
        img (numpy.array): input image of type int or float.
        sigma (float): gaussian noise standard deviation.

    Returns:
        numpy.array: output image with added noise of type float64. Return it without normalizing or clipping it.
    """
    pass
#img = img1
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
#    img normalize_and_scale(img)
    return img1
#    plt.imshow(img1,'gray')
#    plt.imshow(img,'gray')
    
    
#newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
#newImage0 = array(newImage0,dtype=uint8)
#np.max(img)

