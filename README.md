# Automated-Label-Parsing

## Process claim images to

1) Evaluate image file quality: valid image file, not blurred or out of focused, not distorted

2) Apply OCR algorithm to parse model and serial numbers of appliance in claim

## Usage for Pytesseract with EAST detector script
<code>python src/exploratory/pytesseract-with-east-example.py --image "path/to/image/file" --east "path/to/east/file"</code>

For example:
<code>python src/exploratory/pytesseract-with-east-example.py --image "../../images/GE-Model-Tag-Cropped.jpg" --east "../../models/frozen_east_text_detection.pb"</code>

## Usage for preprocessing script
To use the preprocessing script, run the script with the image path:
<code>python src/exploratory/preprocess_image.py --image "path/to/image/file"</code>

For example:
<code>python src/exploratory/preprocess_image.py --image "../images/shadow-image-orig.png"</code>

## Setup steps for Python-tesseract on Windows
(copied from https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i)
1) Install tesseract using windows installer available at: https://github.com/UB-Mannheim/tesseract/wiki
2) Note the tesseract path from the installation. Default installation path at the time of this edit was: C:\Users\USER\AppData\Local\Tesseract-OCR. It may change so please check the installation path.
3) <code>pip install pytesseract</code>
4) Set the tesseract path in the script before calling any Python-tesseract methods, such as <code>image_to_string</code>:
<code>pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'</code>
