import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    ###################################################################################
    # TODO:                                                                           #
    # This function is intended to behave like the scipy.ndimage.filters.correlate    #
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         #
    # of the filter matrix.)                                                          #
    # Your function should work for color images. Simply filter each color            #
    # channel independently.                                                          #
    # Your function should work for filters of any width and height                   #
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       #
    # restriction makes it unambigious which pixel in the filter is the center        #
    # pixel.                                                                          #
    # Boundary handling can be tricky. The filter can't be centered on pixels         #
    # at the image boundary without parts of the filter being out of bounds. You      #
    # should simply recreate the default behavior of scipy.signal.convolve2d --       #
    # pad the input image with zeros, and return a filtered image which matches the   #
    # input resolution. A better approach is to mirror the image content over the     #
    # boundaries for padding.                                                         #
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can #
    # see the desired behavior.                                                       #
    # When you write your actual solution, you can't use the convolution functions    #
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   #
    # Simply loop over all the pixels and do the actual computation.                  #
    # It might be slow.                                                               #
    ###################################################################################
    ###################################################################################
    # NOTE:                                                                           #
    # Some useful functions                                                           #
    #     numpy.pad or numpy.lib.pad                                                  #
    # #################################################################################

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    H_img = R.shape[0]
    W_img = R.shape[1]
    H_filter = imfilter.shape[0]
    W_filter = imfilter.shape[1]
    H_pad = int((H_filter-1)/2)
    W_pad = int((W_filter-1)/2)
    
    npad = ((H_pad, H_pad), (W_pad, W_pad))
    RGB_pad = []
    RGB_pad.append(np.pad(R, pad_width=npad, mode='reflect'))
    RGB_pad.append(np.pad(G, pad_width=npad, mode='reflect'))
    RGB_pad.append(np.pad(B, pad_width=npad, mode='reflect'))
    
    output = np.zeros_like(R)
    
    for each in RGB_pad:
        RGB_new = []
        # convolution of whole matrix
        for m in range(H_img):
            for n in range(W_img):
                # convolution in each small matrix
                total = 0
                total = np.sum(np.multiply(each[m:m+H_filter, n:n+W_filter], imfilter))
                RGB_new.append(total)
                
        RGB_new = np.asarray(RGB_new)
        RGB_new = RGB_new.reshape(H_img, W_img)
        
        # combine RGB channel into 3D array
        output = np.dstack((output, RGB_new))
        
    # remove the zeros array
    output = output[:, :, 1:]
        
#     import scipy.ndimage as ndimage
#     output = np.zeros_like(image)
#     for ch in range(image.shape[2]):
#         output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')
    
    ###################################################################################
    #                                 END OF YOUR CODE                                #
    ###################################################################################
    return output
