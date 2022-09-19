import cv2
import numpy as np

'''
Takes a cv2 image and returns two images: one with shadows removed with normalization
'''
def remove_shadows(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        # Dilate the image, in order to get rid of the text. This step somewhat helps to preserve a bar code.
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        
        # Median blur the result with a decent sized kernel to further suppress any text.
        # This should result in an acceptable background image that contains all the shadows and/or discoloration.
        bg_img = cv2.medianBlur(dilated_img, 21)
        
        # Calculate the difference between the original and the background image just obtained. The bits that are
        # identical will be black (close to 0 difference), the text will be white (large difference).
        # Since black on white is desired, the result is inverted.
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        
        # Normalize the image, so that the full dynamic range is used.
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    
    return result_norm