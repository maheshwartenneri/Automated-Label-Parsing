'''
This exploratory script applies OCR via Pytesseract to the given image
'''

from PIL import Image

import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
# Modify this command based on the path to tesseract application (type may vary by OS)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\bmabb\AppData\Local\Tesseract-OCR\tesseract.exe'

path = r'../../images/model.jpeg'

# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
print(pytesseract.image_to_string(Image.open(path)))