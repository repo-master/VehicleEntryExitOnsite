
'''
TODO:
- Use Sobel or Canny to edge detect and find perspective of plate
'''


import cv2
import numpy as np

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

def crop_rect(img, rect, box):
    angle = rect[2]

    W = rect[1][0]
    H = rect[1][1]
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    if angle < -45:
        angle += 90

    center = ((x1+x2)/2,(y1+y2)/2)
    size = (x2-x1, y2-y1)

    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), float(angle), 1.0)
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))

    return croppedRotated

def compute_skew(src_img):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size

    #print(nlines)
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        #print(ang)
        if np.absolute(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/np.pi

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

from skimage.segmentation import clear_border
from skimage.measure import label as mklabel, regionprops
from skimage.morphology import closing, square
from skimage.filters import gaussian
from skimage.color import label2rgb

def recognize(req, img):
    model = req.app.models['plate_detect_hf_yolos']
    
    #Stage 1: Detect license plate(s?!!)
    res = model.detect(img)
    
    if len(res) == 0: return

    (box, conf, clsid) = res[0]

    #Crop just the plate (with some padding)
    padding = 32
    new_width = 1024 #Can change this

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
    
    V = cv2.split(
        cv2.cvtColor(
            img_raw,
            cv2.COLOR_BGR2HSV)
    )[2]
    
    thresh = threshold_local(V, 51, offset=8, method="gaussian")
    thresh = (V < thresh).astype(np.uint8) * 255
    thresh = cv2.erode(thresh, np.ones((3,3), dtype=np.uint8), iterations=2)
    thresh = cv2.dilate(thresh, np.ones((7,7), dtype=np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((18,18), np.uint8))
    cv2.imshow("b", thresh)
    thresh = cv2.bitwise_not(thresh)
    
    bw = closing(V > thresh, square(3))
    
    cleared = clear_border(bw)
    
    label_image = mklabel(cleared)
    image_label_overlay = label2rgb(label_image, image=img_raw, bg_label=0)
    
    max_size = 1
    cnts_all_regions = []
    for region in regionprops(label_image):
        labelMask = np.zeros(thresh.shape, dtype=np.uint8)
        labelMask[label_image == region.label] = 255
        labelMask = closing(labelMask, square(5))
        cnts, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            eps = 1e-3
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
            
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(approx)
            ca = cv2.contourArea(approx)
            
            keepArea = 8.5e2 < ca < 8e4
            solidity = ca / float(boxW * boxH)
            keepSolidity = solidity > 0.12
            
            aspectRatio = boxW / float(boxH)
            keepAspectRatio = aspectRatio < 1.0
            
            if keepArea and keepSolidity and keepAspectRatio:
                max_size = max(max_size, boxH)
                cnts_all_regions.append((
                    region.label,
                    approx,
                    region.area,
                    ca,
                    region.area/ca,
                    labelMask.copy(),
                    (int(boxX+boxW/2), int(boxY+boxH/2))
                ))

    cnts_quallified = []
    img_stack = []
    if len(cnts_all_regions) > 0:
        mW = max([x[-1][0] for x in cnts_all_regions])
        cnts_all_regions.sort(key=lambda x: (x[0] + x[-1][0] + round(x[-1][1]/max_size)*max_size*mW))

        #Calculate IQR of all areas to get rid of outliers
        cnts_quallified = cnts_all_regions
        if len(cnts_all_regions) > 0:
            areas = np.array([x[2] for x in cnts_all_regions])
            upper_quartile = np.percentile(areas, 75)
            lower_quartile = np.percentile(areas, 25)
            IQR = (upper_quartile - lower_quartile)*0.5
            quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
            cnts_quallified = filter(lambda x: quartileSet[0] <= x[2], cnts_all_regions)

        for lbl, approx, ca, bbarea, density, labelMask, cPos in cnts_quallified:
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        
            cv2.drawContours(image_label_overlay, [box], -1, (0,0,255), 1)
            img_crop = four_point_transform(labelMask, box) #crop_rect(labelMask, rect, box)
            border_pad = 8
            img_chara = img_crop

            if img_chara is not None:
                fit_img = cv2.morphologyEx(img_chara, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                fit_img = cv2.resize(fit_img, (int(max_size*img_chara.shape[1]/img_chara.shape[0]),int(max_size)))
                fit_img = cv2.copyMakeBorder(
                    fit_img,
                    top=border_pad,
                    bottom=border_pad,
                    left=border_pad,
                    right=border_pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )
                img_stack.append(fit_img)

    if len(img_stack) > 0:
        img_stack = np.hstack(img_stack)
        text = pytesseract.image_to_string(img_stack, lang='eng', config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
        cv2.putText(img_raw, text, (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0), thickness=3)
        cv2.putText(img_raw, text, (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,255), thickness=1)
        print('"', text, '"')
        cv2.imshow("B", scaleImgRes(img_stack, scale=0.5)[0])
    
    cv2.imshow("Overlay", scaleImgRes(image_label_overlay, scale=0.5)[0])


    #avg_plate_angle = compute_skew(V)

    '''
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(V, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(V, imgTopHat)
    V = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    blurred = cv2.GaussianBlur(V, (3,3), 15)
    T = threshold_local(blurred, 45, offset=0, method="gaussian")
    thresh = (V > T).astype(np.uint8) * 255
    
    #cv2.imshow("D", blurred)
    #Some cars dont need this
    thresh = cv2.bitwise_not(thresh)
    
    
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, rectKern)
    
    #cv2.imshow("C", tophat)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F,
            dx=1, dy=0, ksize=3)
    gradX = np.absolute(gradX)
    #cv2.imshow("A", V)
    #cv2.imshow("B", gradX)


    labels = measure.label(thresh, background=0)
    m = 1
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype=np.uint8)
        labelMask[labels == label] = 255
        
        #cv2.imshow("ZChar %d" % m, labelMask)
        #m += 1
        
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
                
            if keepArea and keepSolidity and keepAspectRatio:
                hull = cv2.convexHull(c)
                M = cv2.moments(hull)
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
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    masked_box = masked_box[boxY:boxY+boxH,boxX:boxX+boxW]
                    if masked_box.shape[0]*masked_box.shape[1]>0:
                        kernel = np.ones((3,3), np.uint8)  
                        masked_box = cv2.morphologyEx(masked_box, cv2.MORPH_CLOSE, kernel)
                        masked_box = rotate_image(masked_box, avg_plate_angle)
                    
                        text = pytesseract.image_to_string(masked_box, config = '--psm 6').strip()
                        cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,0), thickness=3)
                        cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,255), thickness=1)
                        
                        #cv2.imshow("ZChar %d %s" % (m, text), masked_box)
                        m += 1
    '''
    cv2.imshow("Img", img_raw)
    
    cv2.waitKey(0)
    
