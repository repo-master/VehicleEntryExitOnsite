
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
    
    async def wait_task_timeout(self, task = None, timeout : float = 5.0):
        if task is None: task = self.task
        return self.wait_task_timeout_autocancel(task, timeout, self.logger)

    @staticmethod
    async def wait_task_timeout_autocancel(task : asyncio.Task, timeout : float = 5.0, logger : logging.Logger = None):
        try:
            return await asyncio.wait_for(task, timeout)
        except asyncio.TimeoutError:
            if logger is not None:
                logger.warning("Task was forcibly closed after %.1f seconds timeout" % timeout)
