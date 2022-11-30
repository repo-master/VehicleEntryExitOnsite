
from aiohttp import web
import aiohttp_jinja2
import jinja2

import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage import measure

from datetime import datetime
import statistics
import asyncio
import logging
import typing
import re

from ..components.camera import scaleImgRes
#from ..model import YoloModelCV2
from ..model import YOLOModelTransformers, OCRModelTransformers
from ..model.filter import filter__removeShadow
#from ..model.augmented_filter import augmentation

VEHICLE_DETECT_MODEL_FILE = "models/vehicles.zip"
PLATE_DETECT_MODEL_FILE = "nickmuchi/yolos-small-finetuned-license-plate-detection"
PLATE_DECODE_MODEL_FILE = "microsoft/trocr-small-printed"

NUMBER_PLATE_PATTERN = re.compile(
    r"^(?P<stateCode>[a-z]{2})\s?(?P<rtoCode>[\d{1,2}o])\s?(?P<series>[a-z]{0,2})\s?(?P<vehicleCode>\d{4})$",
    re.MULTILINE | re.IGNORECASE)

class ModelLoader(dict):
    def __init__(self):
        super().__init__()

    async def _load_task(self, model_name : str, loader : typing.Callable, model_path : str):
        self.update({
            model_name: loader(model_path)
        })

    async def load(self):
        await asyncio.gather(*[self._load_task(*x)
            for x in [
                #('vehicle_yolov5', YoloModelCV2.fromZip, VEHICLE_DETECT_MODEL_FILE),
                ('plate_detect_hf_yolos', YOLOModelTransformers.fromHuggingFace, PLATE_DETECT_MODEL_FILE),
                ('plate_decode_hf_trocr', OCRModelTransformers.fromHuggingFace, PLATE_DECODE_MODEL_FILE)
            ]
        ])

async def loadmodel_task_ctx(app : web.Application):
    _log = logging.getLogger('loadmodel_task_ctx')
    _log.debug('Creating task to load models')
    setattr(app, 'models', ModelLoader())
    await app.models.load()
    _log.info('Models loaded')
    yield
    _log.debug('Closing model loader task')

async def decode_image(req : web.Request) -> np.ndarray:
    img_array = np.frombuffer(await req.content.read(), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#Temporary
async def detect(req : web.Request) -> web.Response:
    img = await decode_image(req)
    
    model = req.app.models['vehicle_yolov5']
    res = model.detect(img, zip_results=False, label_str=False, min_score=0.2)
    
    return web.json_response({
        "detection": {
            'bboxes': res[0],
            'detection_scores': res[1],
            'class_ids': res[2]
        }
    })

async def recognize(req : web.Request) -> web.Response:
    img = await decode_image(req)
    
    model = req.app.models['plate_detect_hf_yolos']
    
    #Stage 1: Detect license plate(s?!!)
    res = model.detect(img)
    
    #Image License plate recognition pipeline
    plate_name_candidates = []

    #Iterate over the detections
    for i, (box, conf, clsid) in enumerate(res):
        #Crop just the plate (with some padding)
        padding = 48
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
            continue
            
        #Resize to a standard resolution
        img_aspect = img_cropped.shape[1]/img_cropped.shape[0]
        ##Specify Width. height is automatic
        new_height = int(new_width/img_aspect)

        img_raw = cv2.resize(img_cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        img_raw_noshadow = filter__removeShadow(img_raw)

        #Resize to the new image size
        V = cv2.split(
            cv2.cvtColor(
                img_raw,
                cv2.COLOR_BGR2HSV)
        )[2]
        
        blurred = cv2.bilateralFilter(V, 35, 75, 75)
        T = threshold_local(blurred, 35, offset=15, method="gaussian")
        thresh = (V > T).astype(np.uint8) * 255
        thresh = cv2.bitwise_not(thresh)
        
        edges = cv2.Canny(blurred, 50, 150)
        cv2.imshow("edges", edges)
        
        labels = measure.label(thresh, background=0)
        charCandidates = np.zeros(thresh.shape, dtype=np.uint8)

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
             
        kernel=np.ones((7, 7), np.uint8)
        charCandidates = cv2.dilate(charCandidates, kernel)
        chars_only = cv2.bitwise_and(thresh,thresh, mask=charCandidates)

        #cv2.imshow("Raw", img_raw)
        #cv2.imshow("Thres", thresh)
        #cv2.imshow("Cnt", charCandidates)
        #cv2.imshow("Cnt2", chars_only)

        chars_only = cv2.cvtColor(chars_only, cv2.COLOR_GRAY2BGR)

        #Stage 2: OCR
        generated_text = req.app.models['plate_decode_hf_trocr'].detect(chars_only)
        print(generated_text)
        
        all_plate_matches = [
            #Remove non-alphanumeric characters, and validate against the license plate pattern
            re.sub("\W+", "", x)#NUMBER_PLATE_PATTERN.match(re.sub("\W+", "", x))
            for x in generated_text
        ]
        for x in all_plate_matches:
            plate_name_candidates.append((x, 1.0))
        #all_plate_matches = [x.groupdict() if x else {} for x in all_plate_matches]
        #Histogram
        '''hist = {}
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
                sum([x[1] for x in best_with_parts.values()])/max(1,len(best_with_parts))
            )
            plate_name_candidates.append(best_acc)'''
            

    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(plate_name_candidates) > 0:
        #Send best plate recognition
        best_plt = plate_name_candidates[0]
        return web.json_response({
            "plate_number": best_plt[0],
            "accuracy": best_plt[1],
            "detect_ts": datetime.now(),
            "code": 0
        })
    elif len(res) > 0:
        #Plate detected, but can't recognize
        return web.json_response({
            "message": "Number plate can't be determined",
            "code": 1
        })
    else:
        #No detections
        return web.json_response({
            "message": "No number plate detected",
            "code": 2
        })


@aiohttp_jinja2.template("modeltest.j2.html")
async def tester(request: web.Request) -> typing.Dict[str, str]:
    return {}


def init_routes(app : web.Application):
    #======== For testng page only ========
    app.add_routes([web.static('/static', 'static/')])
    #Setup Jinja template engine
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader('templates/')
    )
    app.router.add_get("/", tester)
    

    #Data models
    app.cleanup_ctx.append(loadmodel_task_ctx)

    #Model APIs
    app.router.add_post("/detect", detect)
    app.router.add_post("/recognize", recognize)
    