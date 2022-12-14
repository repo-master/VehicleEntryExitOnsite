
import re
import typing
from datetime import datetime
import numpy as np

from vehiclebot import imutils
from vehiclebot.types import NUMBER_PLATE_PATTERN

import pickle
from uuid import uuid4

def boxSortCentroidReadingOrder(boxes : np.ndarray) -> np.ndarray:
    boxes = np.array(boxes)
    
    if len(boxes) == 0: return boxes

    #Normalize
    boxes_norm = boxes-np.min(boxes,axis=(0,1))
    boxes_norm = boxes_norm/np.max(boxes_norm,axis=(0,1))
    
    #Sort reading order. Assume boxes are not rotated
    centroids = np.mean(boxes_norm, axis=1)
    sort_order = np.argsort(centroids[:,0]+centroids[:,1])

    return boxes[sort_order]

def recognize(img : np.ndarray, plate_detect_model, text_detect_model, ocr_model) -> typing.Tuple[typing.List[str], float]:
    #Stage 1: Detect license plate
    res = plate_detect_model.detect(img)
    
    if len(res) == 0:
        return None, 0.0 #No plates detected

    (box, conf, clsid) = res[0]

    padding = 32
    new_width = 512

    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))
    img_cropped = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    scale = imutils.scaleImgRes(img_cropped, width=new_width, do_resize=False)
    inv_scale = 1./scale
    pt1 = (int(box[0]-padding*inv_scale), int(box[1]-padding*inv_scale))
    pt2 = (int(box[2]+padding*inv_scale), int(box[3]+padding*inv_scale))
    img_cropped = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    
    boxes = text_detect_model.detect(img_cropped)
    boxes = boxSortCentroidReadingOrder(boxes)
    
    #pickle.dump((img_cropped, boxes), open("img_boxes_%s.pkl" % uuid4().hex, "wb"))
    
    #cv2.imshow("Img plate", img_cropped)
    imgs_text_cropped = []
    for i, box in enumerate(boxes):
        cropped_rotated = imutils.four_point_transform(img_cropped, box)
        #cv2.imshow("Img %d" % i, cropped_rotated)
        imgs_text_cropped.append(cropped_rotated)
        
    if len(imgs_text_cropped) > 0:
        generated_text = ocr_model.detect(imgs_text_cropped)
    else:
        #No text detected????
        return None, conf
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return generated_text, conf

def parse(texts : typing.List[str], conf_plate : float):
    if texts is None:
        return {
            "message": "No number plate detected",
            "code": -2
        }

    acc_all = {"plate_detection": float(conf_plate)}

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
