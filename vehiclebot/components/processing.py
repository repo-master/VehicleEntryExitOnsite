
from vehiclebot.task import AIOTask

import os
import asyncio

class RemoteProcess(AIOTask):
    '''
    Send detection data to a remote service for further processing
    '''

    def __init__(self, tm, task_name, dir : os.PathLike, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.dest_dir = dir
        self.save_queue = asyncio.Queue()

    async def start_task(self):
        if not os.path.isdir(self.dest_dir):
            self.logger.warning("The path \"%s\" does not exist. Creating...")
            os.makedirs(self.dest_dir)

    async def __call__(self):
        await self.save_queue.get()

    async def processDetection(self, image, bbox, **kwargs):
        pass
