# imports
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import imutils
import sys
import cv2
import datetime
import pprint as pp

# local path to tesseract
pytesseract.pytesseract.tesseract_cmd = r'input_local_path_to_tesseract.exe'

# function for preprocessing
def preprocessing(img):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="bottom-to-top")[0]

    mrzBox = None

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        percentWidth = w / float(W)
        percentHeight = h / float(H)
 
        if percentWidth > 0.8 and percentHeight > 0.04:
            mrzBox = (x, y, w, h)
            break

    if mrzBox is None:
        print("[INFO] MRZ could not be found")
        sys.exit(0)
        
    (x, y, w, h) = mrzBox
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.03)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))

    mrz = img[y:y + h, x:x + w]

    return mrz

# function for ocr
def ocr(mrz):
    mrzText = pytesseract.image_to_string(mrz)
    mrzText = mrzText.replace(" ", "")
    return mrzText
    
# function for getting infos
def getInfo(mrzText):
    count = 0
    surname = ""
    name = ""

    for element in range(5, 44):
        if(count == 0):
            if(mrzText[element] != '<'):
                surname += mrzText[element]
            else:
                surname += " "
                if(mrzText[element+1] == '<'):
                    count += 1
                    element+= 1
        if(count == 1):
            if(mrzText[element] != '<'):
                name += mrzText[element]
            else:
                name += " "
                if(mrzText[element+1] == '<'):
                    break

    birth_date = datetime.datetime(
        int("19" + mrzText[58:60]),
        int(mrzText[60:62]), 
        int(mrzText[62:64])
    )
    exp_date = datetime.datetime(
        int("20" + mrzText[66:68]), 
        int(mrzText[68:70]), 
        int(mrzText[70:72])
    )
    issue_date = datetime.datetime(
        int("20" + mrzText[66:68]) - 10, 
        int(mrzText[68:70]), 
        int(mrzText[70:72])
    )
    info = {
        'surname' : surname.strip(),
        'name': name.strip(),
        'passport_no' : mrzText[44:54].strip(),
        'nationality' : mrzText[55:58],
        'sex' : mrzText[65],
        'birth_date' : birth_date,
        'issue_date' : issue_date,
        'exp_date' : exp_date,
    }
    return info

#driver code
image = cv2.imread('passport_1.jpg')
mrz = preprocessing(image)
mrzText = ocr(mrz)
info = getInfo(mrzText)
pp.pprint (info)


