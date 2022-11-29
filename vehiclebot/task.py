
import asyncio
import logging
from typing import Union, Dict, List, Any
from pyee.asyncio import AsyncIOEventEmitter

TaskOrTasks = Union[str, List[str]]

class AIOTask(AsyncIOEventEmitter):
    '''
    Base AIO task class
    '''
    metadata : Dict[str, Any] = {"dependencies": []}
    def __init__(self, tm, task_name : str, *args, **kwargs):
        super().__init__()
        self.tm = tm
        self.task : asyncio.Task = None
        self.name = task_name
        self.logger = logging.getLogger('>'.join([self.__module__, self.__class__.__qualname__, "$"+self.name]))
    async def start_task(self):
        pass
    async def stop_task(self):
        pass
    async def __call__(self):
        '''
        You can do whatever, as long as it is non-blocking and also somehow
        passes control back to asyncio (eg. using sleep, aio socket, etc.)
        '''
        pass
