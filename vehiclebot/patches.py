'''
Apply patches to various built-in modules
'''

import asyncio
from aiortc import rtcdatachannel
from aiohttp import web
import numpy as np
import typing
import humanize

import json
import datetime
from collections.abc import Iterable
import platform
from contextlib import suppress

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ang1 - ang2

def point_angle(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    delta = p1[::-1] - p2[::-1]
    return np.arctan2(*delta)

class JSONifier(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, datetime.timedelta):
            return humanize.precisedelta(obj, format='%.0f')
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Iterable):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
    

async def _e_waitfor_stub(self, timeout=None):
    with suppress(asyncio.TimeoutError, KeyboardInterrupt):
        await asyncio.wait_for(self.wait(), timeout)
    return self.is_set()

async def _q_get_waitfor_stub(self, timeout=None):
    with suppress(asyncio.TimeoutError, KeyboardInterrupt):
        return await asyncio.wait_for(self.get(), timeout)

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

def patch_asyncio_platform_loop_policy():
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())