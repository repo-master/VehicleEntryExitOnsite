
from aiohttp import web
import aiohttp_jinja2
import jinja2

import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import logging
import typing
import traceback
import importlib

from . import recog4 as recognition

from ..model import Model
from ..model.yolodnn import (YOLOModelYOLOv5, YOLOModelTransformers)
from ..model.ocr import OCRModelTransformers
from ..model.craft import CRAFTModel
from ..log import init_root_logger

# Uncomment one of the following to be used for dispatching model inference
# USE_MODEL_POOL = ThreadPoolExecutor
USE_MODEL_POOL = ProcessPoolExecutor

# Where to download and store HuggingFace models
HF_MODEL_CACHEDIR = "models/.transformer-cache/"

COCO_DETECT_MODEL_FILE = "models/external/yolov5n-seg.pt"
VEHICLE_DETECT_MODEL_FILE = "models/external/yolov5-vehicle-best.pt"
VEHICLE_PLATE_DETECT_MODEL_FILE = "models/crossml/vehicle-and-plate-best.pt"
VEHICLE_FB_DETECT_MODEL_FILE = "models/crossml/vehicle-front-back-best.pt"

PLATE_DETECT_MODEL_FILE = "nickmuchi/yolos-small-finetuned-license-plate-detection"
PLATE_DECODE_PROCESSOR_FILE = "microsoft/trocr-small-printed"
PLATE_DECODE_MODEL_FILE = "models/anpr/anpr_demo/"
TEXT_DETECT_MODEL_FILE = "models/craft_mlt_25k.pth"

#Auto-detect device to use. Put 'cpu' or 'cuda' (or cuda:0, etc.) to force a device
DEVICE = None

MODEL_LOADERS = {
    'ultralytics_yolov5_nano': (YOLOModelYOLOv5.fromPT, (COCO_DETECT_MODEL_FILE,), dict(device=DEVICE)),
    'vehicle_yolov5': (YOLOModelYOLOv5.fromPT, (VEHICLE_DETECT_MODEL_FILE,), dict(device=DEVICE)),
    'vehicle_plate_yolov5': (YOLOModelYOLOv5.fromPT, (VEHICLE_PLATE_DETECT_MODEL_FILE,), dict(device=DEVICE)),
    'vehicle_fbk_yolov5': (YOLOModelYOLOv5.fromPT, (VEHICLE_FB_DETECT_MODEL_FILE,), dict(device=DEVICE)),

    'plate_detect_hf_yolos': (YOLOModelTransformers.fromHuggingFace, (PLATE_DETECT_MODEL_FILE,), dict(cache_dir=HF_MODEL_CACHEDIR, device=DEVICE)),
    'plate_decode_hf_trocr': (OCRModelTransformers.fromHuggingFace, (PLATE_DECODE_MODEL_FILE,), dict(processor_path=PLATE_DECODE_PROCESSOR_FILE, cache_dir=HF_MODEL_CACHEDIR, device=DEVICE)),
    'text_detect_pt_craft': (CRAFTModel.fromCRAFT, (TEXT_DETECT_MODEL_FILE,), dict(device=DEVICE))
}

PRELOAD_MODELS = []

#Data load helpers

def decode_image(img_buf) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(img_buf, dtype=np.uint8), cv2.IMREAD_COLOR)

def get_model(model_name) -> Model:
    if not globals().get(model_name, None):
        print("Loading model %s" % model_name)
        mdl_loader = MODEL_LOADERS[model_name]
        model = mdl_loader[0](*mdl_loader[1], **mdl_loader[2])
        globals().update({model_name: model})
    return globals().get(model_name)

def models_preload(model_list : list):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    log = init_root_logger(logging.getLogger(__name__))
    log.info("Preloading models [%s]...", ','.join(model_list))
    try:
        n_models = list(map(get_model, model_list))
    except:
        log.exception("Error preloading")
    log.info("%d Models preloaded", len(n_models))

#Tasks for workers

def detect_task(img_buf, model_detector):
    model = get_model(model_detector)
    return model.detect(decode_image(img_buf), min_confidence=0.4)

def recognize_task(img_buf, ocr_model):
    #importlib.reload(recognition)
    ocr_model_o = get_model(ocr_model)
    return recognition.parse(
        recognition.recognize(
            decode_image(img_buf),
            ocr_model_o
        )
    )


#Routes

async def detect(req : web.Request) -> web.Response:
    '''Detect anything'''
    
    res = await asyncio.get_running_loop().run_in_executor(
        req.app['pool_detect'],
        detect_task,
        await req.content.read(),
        req.query.get('model', 'ultralytics_yolov5_nano')
    )

    return web.json_response({
        'detection': {
            'bboxes': res[0],
            'detection_scores': res[1],
            'class_ids': res[2]
        }
    })

async def recognize(req : web.Request) -> web.Response:
    try:
        res = await asyncio.get_running_loop().run_in_executor(
            req.app['pool_recognize'],
            recognize_task,
            await req.content.read(),
            'plate_decode_hf_trocr'
        )
        return web.json_response(res)
    except:
        return web.Response(text=traceback.format_exc(), status=500)
    

@aiohttp_jinja2.template("modeltest.j2.html")
async def tester(request: web.Request) -> typing.Dict[str, str]:
    return {}

def init_routes(app : web.Application):
    log = init_root_logger(logging.getLogger(__name__))
    #======== For testng page only ========
    app.add_routes([web.static('/static', 'static/')])
    #Setup Jinja template engine
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader('templates/')
    )
    app.router.add_get("/", tester)
    
    log.info("Starting worker pool")
    app['pool_detect'] = ProcessPoolExecutor(max_workers=2, initializer=models_preload, initargs=(PRELOAD_MODELS,))
    app['pool_recognize'] = ProcessPoolExecutor(max_workers=2, initializer=models_preload, initargs=(PRELOAD_MODELS,))

    #Model APIs
    app.router.add_post("/detect", detect)
    app.router.add_post("/recognize", recognize)
    