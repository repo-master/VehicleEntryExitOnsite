
from vehiclebot.task import AIOTask

import asyncio
import typing
import cv2

from concurrent.futures import ProcessPoolExecutor

class VideoDisplayProcess:
    @staticmethod
    def imshow(window_name, img):
        cv2.imshow(window_name, img)
    
    @staticmethod
    def destroyAllWindows():
        cv2.destroyAllWindows()
    
    @staticmethod
    def waitKey(delay):
        return cv2.waitKey(delay)

class VideoDisplay(AIOTask):
    def __init__(self, tm, task_name, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self._stop = asyncio.Event()
        self.proc = ProcessPoolExecutor(max_workers=1)
    
    def call_process(self, func, *args) -> asyncio.Future:
        task = asyncio.get_event_loop().run_in_executor(self.proc, func, *args)
        task.add_done_callback(self._proc_done_callback)
        return task

    def _proc_done_callback(self, future : asyncio.Future):
        exc = future.exception()
        if exc:
            self.logger.exception(exc)
    
    async def stop_task(self):
        self.logger.info("Stopping video display")
        self._stop.set()
        await self.task

    async def __call__(self):
        while not self._stop.is_set():
            #Poll window manager
            key = await self.call_process(VideoDisplayProcess.waitKey, 15)

            #if key == ord('q'):
            #    self._stop = True
            #    break
        await self.call_process(VideoDisplayProcess.destroyAllWindows)
    
    async def imshow(self, window_name, img):
        return await self.call_process(VideoDisplayProcess.imshow, window_name, img)

