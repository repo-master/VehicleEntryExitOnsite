


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
        

blurred = cv2.bilateralFilter(V, 15, 75, 75)
T = threshold_local(blurred, 15, offset=15, method="gaussian")
thresh = (V > T).astype(np.uint8) * 255
thresh = cv2.bitwise_not(thresh)

labels = measure.label(thresh, background=0)
charCandidates = np.zeros(thresh.shape, dtype=np.uint8)

matched_result = []
for label in np.unique(labels):
    if label == 0:
        continue
    labelMask = np.zeros(thresh.shape, dtype=np.uint8)
    labelMask[labels == label] = 255
            
    cnts, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(labelMask, [box], -1, 255, 2)
                
        '''src_pts = box.astype("float32")
        dst_pts = np.array([
                [0, 0],
                [width-1, 0],
                [width-1, height-1],
                [0, height-1]], dtype="float32")
        #M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        #warped = cv2.warpPerspective(thresh, M, (width, height))
        #cropped_rect = warped#thresh[boxY:boxY+boxH,boxX:boxX+boxW]
        #cv2.imshow("C_w %d" % aaa, cv2.resize(cropped_rect, (200, 200)))
        '''

        aspectRatio = boxW / float(boxH)
        solidity = cv2.contourArea(c) / float(boxW * boxH)
        heightRatio = boxH / float(img_raw.shape[0])

        keepArea = cv2.contourArea(c) > 20
        keepAspectRatio = aspectRatio < 1.0
        keepSolidity = solidity > 0.2
        keepHeight = heightRatio > 0.4 and heightRatio < 0.95
                
        if keepArea and keepAspectRatio and keepSolidity:
            hull = cv2.convexHull(c)
            cv2.drawContours(charCandidates, [hull], -1, 255, -1)
            matched_result.append({
                'contour': hull,
                'x': boxX,
                'y': boxY,
                'w': boxW,
                'h': boxH,
                'cx': boxX + (boxW / 2),
                'cy': boxY + (boxH / 2)
            })
             
kernel=np.ones((7, 7), np.uint8)
charCandidates = cv2.dilate(charCandidates, kernel)
chars_only = cv2.bitwise_and(thresh,thresh, mask=charCandidates)
        
PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []
        
sorted_chars = sorted(matched_result, key=lambda x: x['cx'])
        
plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        
sum_height = 0
for d in sorted_chars:
    sum_height += d['h']

plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
triangle_hypotenus = np.linalg.norm(
    np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
    np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
)
    
angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
img_rotated = cv2.warpAffine(thresh, M=rotation_matrix, dsize=(new_width, new_height))
    
img_cropped = cv2.getRectSubPix(
    img_rotated, 
    patchSize=(int(plate_width), int(plate_height)), 
    center=(int(plate_cx), int(plate_cy))
)
    
if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
    continue
    
plate_imgs.append(img_cropped)
plate_infos.append({
    'x': int(plate_cx - plate_width / 2),
    'y': int(plate_cy - plate_height / 2),
    'w': int(plate_width),
    'h': int(plate_height)
})
cv2.imshow("Img", img_cropped)

#Stage 2: OCR
cv2.imshow("Test1", V)
cv2.imshow("Test2", chars_only)
cv2.imshow("Test3", thresh)
cv2.imshow("Test3", charCandidates)
cv2.imshow("Test4", blurred)
cv2.waitKey(0)
        
print("1", pytesseract.image_to_string(V))
print("2", pytesseract.image_to_string(chars_only))
print("3", pytesseract.image_to_string(thresh))
print("4", pytesseract.image_to_string(charCandidates))
print("5", pytesseract.image_to_string(blurred))
        
generated_text = []
all_plate_matches = [
    #Remove non-alphanumeric characters, and validate against the license plate pattern
    NUMBER_PLATE_PATTERN.match(re.sub("\W+", "", x))
    for x in generated_text
]
all_plate_matches = [x.groupdict() if x else {} for x in all_plate_matches]
#Histogram
hist = {}
for i in all_plate_matches:
    for j,k in i.items():
        if j not in hist:
            hist[j] = {}
        if len(k) == 0: continue
        if k not in hist[j]:
            hist[j][k] = 1
        else:
            hist[j][k] += 1
                        
best_with_parts = {k : max(v.items(), key=lambda x: x[1]) for k,v in hist.items()}
if len(best_with_parts) > 0:
    best_acc = (
        ' '.join([x[0] for x in best_with_parts.values()]),
        sum([x[1] for x in best_with_parts.values()])/MAX_AUGS/max(1,len(best_with_parts))
    )
    plate_name_candidates.append(best_acc)
            