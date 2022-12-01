import cv2
import numpy as np

import uuid
import pytesseract
from skimage.filters import threshold_local
from skimage import measure
from scipy.spatial import distance as dist

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def scaleImgRes(img : np.ndarray, scale : float = None, width : float = None, height : float = None, do_resize : bool = True):
    old_h, old_w = img.shape[:2]

    if height is not None:
        scale = height/old_h
    elif width is not None:
        scale = width/old_w
    if scale is None:
        scale = 1.0
    new_w = int(old_w * scale)
    new_h = int(old_h * scale)

    if do_resize:
        return cv2.resize(img, (new_w, new_h)), scale
    return scale


def recognize(req, img):
    model = req.app.models['plate_detect_hf_yolos']
    
    #Stage 1: Detect license plate(s?!!)
    res = model.detect(img)
    
    if len(res) == 0: return

    (box, conf, clsid) = res[0]

    #Crop just the plate (with some padding)
    padding = 16
    new_width = 512 #Can change this

    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))
    img_cropped = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    scale = scaleImgRes(img_cropped, width=new_width, do_resize=False)
    inv_scale = 1./scale
    pt1 = (int(box[0]-padding*inv_scale), int(box[1]-padding*inv_scale))
    pt2 = (int(box[2]+padding*inv_scale), int(box[3]+padding*inv_scale))
    img_cropped = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]

    #if image detection was incorrect
    if img_cropped.shape[0]*img_cropped.shape[1] == 0:
        return
            
    #Resize to a standard resolution
    img_aspect = img_cropped.shape[1]/img_cropped.shape[0]
    new_height = int(new_width/img_aspect)

    img_raw = cv2.resize(img_cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite("%s.png" % uuid.uuid4().hex, img_raw)

    V = cv2.split(
        cv2.cvtColor(
            img_raw,
            cv2.COLOR_BGR2HSV)
    )[2]

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(V, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(V, imgTopHat)
    V = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    blurred = cv2.bilateralFilter(V, 100, 75, 75)
    T = threshold_local(blurred, 45, offset=0, method="gaussian")
    thresh = (V > T).astype(np.uint8) * 255
    cv2.imshow("Thresh_blur", blurred)
    cv2.imshow("Thresh_local", thresh.copy())
    
    #Some cars dont need this
    thresh = cv2.bitwise_not(thresh)
    

    labels = measure.label(thresh, background=0)
    charMasks = np.zeros(thresh.shape, dtype=np.uint8)
    charCandidates = np.zeros(thresh.shape, dtype=np.uint8)
    m = 1
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype=np.uint8)
        labelMask[labels == label] = 255
        
        cnts, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #TODO: Calculate average of all contours, find contours falling in that range

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(img_raw.shape[0])

            keepArea = cv2.contourArea(c) > 20 and cv2.contourArea(c) < 5e4
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.2
            keepHeight = heightRatio > 0.4 and heightRatio < 0.95
                
            if keepArea and keepSolidity:#keepArea and keepAspectRatio and keepSolidity:
                hull = cv2.convexHull(c)
                charMask = np.zeros(thresh.shape, dtype=np.uint8)
                cv2.drawContours(charMask, [hull], -1, 255, -1)
                masked_box = cv2.bitwise_and(thresh, thresh, mask=charMask)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(hull)
                pd = 4
                boxX -= pd
                boxY -= pd
                boxW += 2*pd
                boxH += 2*pd
                boxX = max(boxX, 0)
                boxY = max(boxY, 0)
                masked_box = masked_box[boxY:boxY+boxH,boxX:boxX+boxW]
                if masked_box.shape[0]*masked_box.shape[1]>0:
                    cv2.imshow("ZChar %d" % m, masked_box)
                    m += 1
                
                cv2.drawContours(charMasks, [hull], -1, 255, -1)

             
    plate_img = thresh
    kernel=np.ones((9, 9), np.uint8)
    charCandidatesM = cv2.erode(charMasks, kernel,iterations=2)
    charCandidatesM = cv2.dilate(charCandidatesM, kernel)
    cnts, _ = cv2.findContours(charCandidatesM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cs = sorted(cnts, key=lambda x:0.01*x[0,0,0]-cv2.contourArea(x))[:10]
        cv2.drawContours(charCandidates, cs, -1, 255, -1)

    #cv2.imshow("CC", charCandidates.copy())
    charCandidates = cv2.morphologyEx(charCandidates, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65,65)))
    
    #cv2.imshow("CC2", charCandidates.copy())
    cnts2, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts2) > 0:
        c = max(cnts2, key=cv2.contourArea)
        #(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        translated_box = box - np.mean(box, axis=0)
        scaled_box = translated_box*1.12
        retranslated_box = scaled_box + np.mean(box, axis=0)
        box = np.int0(retranslated_box)
        
        cv2.drawContours(img_raw, [box], -1, (0,0,255), 2)
        plate_img = cv2.bitwise_and(plate_img, plate_img, mask=charMasks)
        plate_img = four_point_transform(plate_img, box)

    #chars_only = cv2.bitwise_and(plate_img, plate_img, mask=charCandidates)
    #kernel=np.ones((3, 1), np.uint8)
    #chars_only = cv2.dilate(x, kernel)
    
    #cv2.imshow("Test", img_raw)
    #cv2.imshow("Test2", plate_img)
    
    text = pytesseract.image_to_string(plate_img,config = '--psm 6')
    print("rec",text)
    cv2.waitKey(0)

def recognize2(img):
    V = cv2.split(
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV)
    )[2]

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(V, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(V, imgTopHat)
    V = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    blurred = cv2.bilateralFilter(V, 20, 75, 75)
    T = threshold_local(blurred, 45, offset=0, method="gaussian")
    thresh = (V > T).astype(np.uint8) * 255
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    pts = []
    def click_and_crop(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            pts.append((x, y))
            print("added", x, y)
    cv2.namedWindow("Test2")
    cv2.setMouseCallback("Test2", click_and_crop)
    cv2.imshow("Test2", V)
    cv2.waitKey(0)
    V = four_point_transform(thresh, np.array(pts))
    cv2.imshow("Test2", V)
    
    text = pytesseract.image_to_string(V,config ='--psm 6')
    print(text)
    cv2.waitKey(0)

