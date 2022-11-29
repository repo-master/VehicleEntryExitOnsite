'''
Apply patches to various built-in modules
'''

import asyncio
from aiortc import rtcdatachannel
from aiohttp import web
import numpy as np

import json
import datetime

class JSONifier(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

async def _e_waitfor_stub(self, timeout=None):
    try:
        await asyncio.wait_for(self.wait(), timeout)
    except asyncio.TimeoutError:
        pass
    finally:
        return self.is_set()

async def _q_get_waitfor_stub(self, timeout=None):
    try:
        return await(await asyncio.wait_for(self.get(), timeout))
    except asyncio.TimeoutError:
        pass

def asyncio_monkey_patch():
    setattr(asyncio.Event, 'wait_for', _e_waitfor_stub)
    setattr(asyncio.Queue, 'get_wait_for', _q_get_waitfor_stub)
    
def patch_aiortc_datachannel_json():
    #Reference to old send
    old_send = rtcdatachannel.RTCDataChannel.send
    def send_json(self, data, *args, **kwargs):
        jsonified = json.dumps(data, cls=JSONifier)
        return old_send(self, jsonified, *args, **kwargs)
    rtcdatachannel.RTCDataChannel.send = send_json

def patch_aiohttp_json_response():
    old_json_response = web.json_response
    def json_response(*args, **kwargs):
        return old_json_response(dumps=lambda data: json.dumps(data, cls=JSONifier), *args, **kwargs)
    web.json_response = json_response
    
def aiortc_monkey_patch():
    patch_aiortc_datachannel_json()

def aiohttp_monkey_patch():
    patch_aiohttp_json_response()