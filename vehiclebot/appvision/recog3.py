
import re
import typing
from datetime import datetime
import numpy as np

from vehiclebot import imutils
from vehiclebot.types import NUMBER_PLATE_PATTERN

import pickle
from uuid import uuid4

def boxSortCentroidReadingOrder(boxes : np.ndarray) -> np.ndarray:
    if boxes is None or len(boxes) == 0: return boxes

    boxes = np.array(boxes)

    #Normalize
    boxes_norm = boxes-np.min(boxes,axis=(0,1))
    boxes_norm = boxes_norm/np.max(boxes_norm,axis=(0,1))
    
    #Sort reading order. Assume boxes are not rotated
    centroids = np.mean(boxes_norm, axis=1)
    sort_order = np.argsort(centroids[:,0]+centroids[:,1])

    return boxes[sort_order]

def recognize(img : np.ndarray, text_detect_model, ocr_model) -> typing.List[str]:
    new_width = 512

    img_scaled, scale = imutils.scaleImgRes(img, width=new_width)
    boxes = text_detect_model.detect(img_scaled)

    if boxes is None:
        return #Detect error

    boxes = boxSortCentroidReadingOrder(boxes)

    imgs_text_cropped = []
    for box in boxes:
        cropped_rotated = imutils.four_point_transform(img_scaled, box)
        imgs_text_cropped.append(cropped_rotated)

    if len(imgs_text_cropped) == 0:
        #No text detected????
        return

    generated_text = ocr_model.detect(imgs_text_cropped)
    return generated_text

def parse(texts : typing.List[str]):
    if texts is None:
        return {
            "message": "No number plate detected",
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
    
    plate_text_parsed = NUMBER_PLATE_PATTERN.match(full_text)
    if plate_text_parsed is None:
        return {
            "message": "Number plate can't be determined",
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
