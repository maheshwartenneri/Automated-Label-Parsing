import argparse
import os
import cv2

from remove_shadows import remove_shadows

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, help="path to input image")
    ap.add_argument("-t", "--transf", type=str, help="transformations to apply: r(otation), c(rop), ")
    args = vars(ap.parse_args())
    
    image_path = args["image"]
    base_file_name = os.path.basename(image_path)
    
    img = cv2.imread(image_path)
    result_norm = remove_shadows(img)
    
    cv2.imwrite('../images/preprocessed_' + base_file_name, result_with_norm)
    