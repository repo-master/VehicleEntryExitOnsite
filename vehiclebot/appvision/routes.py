
from aiohttp import web
import aiohttp_jinja2
import jinja2

import cv2
import numpy as np

from datetime import datetime
from concurrent.futures.thread import ThreadPoolExecutor
import asyncio
import logging
import typing
import traceback
import importlib

from . import recog3 as recognition

from ..model.yolodnn import (YOLOModelCV2, YOLOModelTransformers)
from ..model.ocr import OCRModelTransformers
from ..model.craft import CRAFTModel
from ..log import init_root_logger

VEHICLE_DETECT_MODEL_FILE = "models/vehicles.zip"
PLATE_DETECT_MODEL_FILE = "nickmuchi/yolos-small-finetuned-license-plate-detection"
PLATE_DECODE_MODEL_FILE = "microsoft/trocr-small-printed"
TEXT_DETECT_MODEL_FILE = "models/craft_mlt_25k.pth"

MODEL_LOADERS = {
    'vehicle_yolov5': (YOLOModelCV2.fromZip, (VEHICLE_DETECT_MODEL_FILE,), {}),
    'plate_detect_hf_yolos': (YOLOModelTransformers.fromHuggingFace, (PLATE_DETECT_MODEL_FILE,), dict(cache_dir="models/.transformer-cache/")),
    'plate_decode_hf_trocr': (OCRModelTransformers.fromHuggingFace, (PLATE_DECODE_MODEL_FILE,), dict(cache_dir="models/.transformer-cache/")),
    'text_detect_pt_craft': (CRAFTModel.fromCRAFT, (TEXT_DETECT_MODEL_FILE,), {})
}

PRELOAD_MODELS = ['vehicle_yolov5']

#Data load helpers

def decode_image(img_buf) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(img_buf, dtype=np.uint8), cv2.IMREAD_COLOR)

def get_model(model_name):
    if not globals().get(model_name, None):
        mdl_loader = MODEL_LOADERS[model_name]
        model = mdl_loader[0](*mdl_loader[1], **mdl_loader[2])
        globals().update({model_name: model})
    return globals().get(model_name)

def models_preload(model_list : list):
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
    return model.detect(decode_image(img_buf), zip_results=False, label_str=False, min_score=0.2)

def recognize_task(img_buf, detect_model, text_detect_model, ocr_model):
    importlib.reload(recognition)
    detect_model_o = get_model(detect_model)
    text_detect_model_o = get_model(text_detect_model)
    ocr_model_o = get_model(ocr_model)
    return recognition.parse(
        *recognition.recognize(
            decode_image(img_buf),
            detect_model_o,
            text_detect_model_o,
            ocr_model_o
        )
    )


#Routes

async def detect(req : web.Request) -> web.Response:
    '''Detect anything'''
    
    res = await asyncio.get_running_loop().run_in_executor(
        req.app['pool'],
        detect_task,
        await req.content.read(),
        req.query.get('model', 'vehicle_yolov5')
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
            req.app['pool'],
            recognize_task,
            await req.content.read(),
            'plate_detect_hf_yolos',
            'text_detect_pt_craft',
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
    app['pool'] = ThreadPoolExecutor(max_workers=2, initializer=models_preload, initargs=(PRELOAD_MODELS,))

    #Model APIs
    app.router.add_post("/detect", detect)
    app.router.add_post("/recognize", recognize)
    