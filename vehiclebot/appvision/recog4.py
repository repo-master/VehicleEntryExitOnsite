
import re
import typing
from datetime import datetime
import numpy as np

from vehiclebot import imutils
from vehiclebot.types import NUMBER_PLATE_PATTERN

import cv2

from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=1, limit=15):
    '''
    Correct image skew in 2D space (rotation).

    @author: tapan_anant
    @copyright: CrossML 2023

    '''
    # image = cv2.imread(image)
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def recognize(img : np.ndarray, ocr_model) -> typing.List[str]:
    #Scale to a fixed size, multiple of 16 (as TrOCR block size is 16x16)
    img_scaled, _ = imutils.scaleImgRes(img, height=96)
    final_image = correct_skew(img_scaled)
    final_image, _ = imutils.scaleImgRes(final_image, height=64)
    #final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    #final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
    #cv2.imshow("Test", final_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    generated_text = ocr_model.detect(final_image)
    return generated_text

def parse(texts : typing.List[str]):
    if texts is None:
        return {
            "message": "No text detected",
            "code": -2
        }

    acc_all = {}

    full_text = re.sub('[^\w\s]+', '', ' '.join(texts))
    
    #TODO:
    '''
    for num in range(len(texts), 1, -1):
        s = ' '.join(texts[:num])
        r = NUMBER_PLATE_PATTERN.match(s)
        print(s, '-', r.groupdict() if r is not None else 'Nope')
    '''
    
    plate_text_parsed = NUMBER_PLATE_PATTERN.search(full_text)
    if plate_text_parsed is None:
        return {
            "message": "Text detected does not fit in the plate format",
            "code": 1,
            "plate_number": {},
            "plate_str": None,
            "plate_raw": full_text,
            "detect_ts": datetime.now(),
            "accuracy": acc_all
        }
    
    return {
        "code": 0,
        "plate_number": plate_text_parsed.groupdict(),
        "plate_str": ' '.join(plate_text_parsed.groups()),
        "plate_raw": full_text,
        "detect_ts": datetime.now(),
        "accuracy": acc_all
    }
