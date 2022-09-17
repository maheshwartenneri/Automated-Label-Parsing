'''
This exploratory script crops the image given in the path so that just the relevant text is kept

Press any key to exit
'''

# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

import cv2

path = r'../../images/transf-image-orig.png'

img = cv2.imread(path)

# crop_img = img[y:y+h, x:x+w]
crop_img = img[500:500+400, 50:50+500]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)