# https://medium.com/analytics-vidhya/targeted-ocr-on-documents-with-opencv-and-pytesseract-edc10b5ecb62
# https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
# https://arxiv.org/abs/1704.03155
# https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/blob/master/frozen_east_text_detection.pb

from imutils.object_detection import non_max_suppression
from PIL import Image
import cv2
import math
import argparse
import pytesseract
import numpy as np

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)
    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\bmabb\AppData\Local\Tesseract-OCR\tesseract.exe'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
# print(pytesseract.image_to_string(Image.open('GE-Model-Tag-Cropped.jpg')))

img = cv2.imread(args["image"])
# img = cv2.imread('GE-Model-Tag-Cropped.jpg',0)
orig = img.copy()
(origH, origW) = img.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)
# resize the image and grab the new image dimensions
image = cv2.resize(img, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested in -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    dX = int((endX - startX) * args["padding"])
    dY = int((endY - startY) * args["padding"])
    # apply padding to each side of the bounding box, respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))
    # extract the actual padded ROI
    roi = orig[startY:endY, startX:endX]

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 1, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
    # add the bounding box coordinates and OCR'd text to the list
    # of results
    results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])
# loop over the results
for ((startX, startY, endX, endY), text) in results:
	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)
    
'''
img_copy = img.copy()
img_canny = cv2.Canny(img_copy, 50, 100, apertureSize = 3)
img_hough = cv2.HoughLinesP(img_canny, 1, math.pi / 180, 100, minLineLength = 100, maxLineGap = 10)

(x, y, w, h) = (np.amin(img_hough, axis = 0)[0,0], np.amin(img_hough, axis = 0)[0,1], np.amax(img_hough, axis = 0)[0,0] - np.amin(img_hough, axis = 0)[0,0], np.amax(img_hough, axis = 0)[0,1] - np.amin(img_hough, axis = 0)[0,1])
img_roi = img_copy[y:y+h,x:x+w]

img_roi = cv2.rotate(img_roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
(height, width) = img_roi.shape
img_roi_copy = img_roi.copy()
dim_mrz = (x, y, w, h) = (1, round(height*0.9), width-3, round(height-(height*0.9))-2)
img_roi_copy = cv2.rectangle(img_roi_copy, (x, y), (x + w ,y + h),(0,0,0),2)

img_mrz = img_roi[y:y+h, x:x+w]
img_mrz =cv2.GaussianBlur(img_mrz, (3,3), 0)
ret, img_mrz = cv2.threshold(img_mrz,127,255,cv2.THRESH_TOZERO)

mrz = pytesseract.image_to_string(img_mrz, config = '--psm 12')
'''
# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
#print(mrz)

# List of available languages
# print(pytesseract.get_languages(config=''))

# French text image to string
#print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# Batch processing with a single file containing the list of multiple image file paths
#print(pytesseract.image_to_string('images.txt'))

# Timeout/terminate the tesseract job after a period of time
#try:
#    print(pytesseract.image_to_string('test.jpg', timeout=2)) # Timeout after 2 seconds
#   print(pytesseract.image_to_string('test.jpg', timeout=0.5)) # Timeout after half a second
#except RuntimeError as timeout_error:
#   # Tesseract processing is terminated
#   pass

# Get bounding box estimates
#print(pytesseract.image_to_boxes(Image.open('test.png')))

# Get verbose data including boxes, confidences, line and page numbers
#print(pytesseract.image_to_data(Image.open('test.png')))

# Get information about orientation and script detection
#print(pytesseract.image_to_osd(Image.open('test.png')))

# Get a searchable PDF
#pdf = pytesseract.image_to_pdf_or_hocr('test.png', extension='pdf')
#with open('test.pdf', 'w+b') as f:
#    f.write(pdf) # pdf type is bytes by default

# Get HOCR output
#hocr = pytesseract.image_to_pdf_or_hocr('test.png', extension='hocr')

# Get ALTO XML output
#xml = pytesseract.image_to_alto_xml('test.png')