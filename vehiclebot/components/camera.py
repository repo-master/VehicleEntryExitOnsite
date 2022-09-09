
from vehiclebot.task import AIOTask
from vehiclebot.multitask import GlobalInstances

import asyncio
import logging
import typing
import cv2

from concurrent.futures import ProcessPoolExecutor

#Synchronous code in isolated process
#This is needed because OpenCV function calls are not async
class CameraSourceProcess(GlobalInstances):
    @classmethod
    def start_capture(cls, src : typing.Union[int, str]) -> str:
        return cls.create_instance(cv2.VideoCapture(src))

    @classmethod
    def stop_capture(cls, instance : str):
        cap = cls.get_instance(instance)
        cap.release()

    @classmethod
    def read_frame(cls, instance : str):
        cap = cls.get_instance(instance)
        return cap.read()
    
    @classmethod
    def skip_frames(cls, instance : str, frames : int):
        self = cls.get_instance(instance)
        for _ in range(frames):
            self.read()
    
class CameraSource(AIOTask):
    def __init__(self, tm, task_name, src : typing.Union[int, str], skip_frames : int = 0, show_output : bool = False, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.source = src
        self._skipframes = skip_frames
        if self._skipframes < 0: self._skipframes = 0
        self.show_output = show_output

        self.cap = None
        self._latest_frame = None
        self.cap_get_wait = asyncio.Event()

        self._stop = False
        self.proc = ProcessPoolExecutor(max_workers=1, initializer=CameraSourceProcess.init)
    
    def call_process(self, func, *args):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.proc, func, *args)
    
    async def start_task(self):
        self.logger.info("Starting video capture of source \"%s\"" % str(self.source))
        try:
            self.cap = await self.call_process(CameraSourceProcess.start_capture, self.source)
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.cap_get_wait.set()

    async def stop_task(self):
        self._stop = True
        await self.task
        if self.cap is None: return
        self.logger.info("Stopping video capture")
        try:
            await self.call_process(CameraSourceProcess.stop_capture, self.cap)
        except Exception as e:
            self.logger.exception(e)

    async def __call__(self):
        await self.cap_get_wait.wait()
        if self.cap is None: return

        #Skip frames
        await self.call_process(CameraSourceProcess.skip_frames, self.cap, self._skipframes)

        while not self._stop:
            #Fetch images as fast as possible
            ret, img = await self.call_process(CameraSourceProcess.read_frame, self.cap)
            if ret:
                #img = cv2.resize(img, (800,450))
                self._latest_frame = img
                if self.show_output:
                    await self.tm['videodisplay'].imshow("Video", img)
            else:
                break

    @property
    def frame(self):
        return self._latest_frame
