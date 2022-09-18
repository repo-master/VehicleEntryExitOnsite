'''
Apply patches to various built-in modules
'''

import asyncio

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
