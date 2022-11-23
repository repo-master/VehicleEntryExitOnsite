'''
Apply patches to various built-in modules
'''

import asyncio
from aiortc import rtcdatachannel

import json
import datetime

class JSONifier(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("date format", flush=True)
            return obj.isoformat()
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
        jsonified = json.dumps(data, cls=JSONifier, default=str)
        return old_send(self, jsonified, *args, **kwargs)
    rtcdatachannel.RTCDataChannel.send = send_json

def aiortc_monkey_patch():
    patch_aiortc_datachannel_json()
