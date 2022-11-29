
from aiohttp import web
import aiohttp_jinja2
import jinja2

import cv2
import numpy as np

from datetime import datetime
import asyncio
import logging
import typing
import re

from ..model import YoloModelCV2
from ..model import YOLOModelTransformers, OCRModelTransformers
from ..model.augmented_filter import augmentation

VEHICLE_DETECT_MODEL_FILE = "models/vehicles.zip"
PLATE_DETECT_MODEL_FILE = "nickmuchi/yolos-small-finetuned-license-plate-detection"
PLATE_DECODE_MODEL_FILE = "microsoft/trocr-small-printed"

NUMBER_PLATE_PATTERN = re.compile(
    r"^(?P<stateCode>[a-z]{2})\s?(?P<rtoCode>\d{1,2})\s?(?P<series>[a-z]{0,2})\s?(?P<vehicleCode>\d{4})$",
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
                ('vehicle_yolov5', YoloModelCV2.fromZip, VEHICLE_DETECT_MODEL_FILE),
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
        padding = 0
        pt1 = (int(box[0]-padding), int(box[1]-padding))
        pt2 = (int(box[2]+padding), int(box[3]+padding))
        img_cropped = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
            
        #if image detection was incorrect
        if img_cropped.shape[0]*img_cropped.shape[1] == 0:
            continue
            
        #Resize to a standard resolution
        img_aspect = img_cropped.shape[1]/img_cropped.shape[0]
        ##Specify Width. height is automatic
        new_width = 512 #Can change this
        new_height = int(new_width/img_aspect)

        #GRAYSCALE, Resize to the new image size
        img_raw = cv2.cvtColor(
            cv2.resize(img_cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_BGR2GRAY
        )

        MAX_AUGS = 12
        imgs_aug = augmentation(img_raw, num_iters = MAX_AUGS, aug_max_combinations=0)
        #BGR for text detection
        imgs_aug = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in imgs_aug]
            
        #Stage 2: OCR
        generated_text = req.app.models['plate_decode_hf_trocr'].detect(imgs_aug)
            
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
    