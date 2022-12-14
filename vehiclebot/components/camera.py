
from vehiclebot.task import AIOTask, TaskOrTasks
from vehiclebot.multitask import AsyncProcess
from vehiclebot.imutils import scaleImgRes

import time
import asyncio
import threading
import typing

import cv2
import numpy as np

#Synchronous code in isolated process
#This is needed because OpenCV function calls are not async
class CameraSourceProcess(threading.Thread):
    def __init__(self, exception_mode : bool = False):
        super().__init__(daemon=True)
        self._stopEv = threading.Event()
        self._frame : np.ndarray = None
        self._enableCapture = False
        self._update_rate : float = 1000

        self.cap = cv2.VideoCapture()
        self.setExceptionMode(exception_mode)

        self.start()

    def stop(self, timeout : float = None):
        self._stopEv.set()
        self.join(timeout=timeout)

    def cleanup(self):
        self.close()

    def run(self):
        next_time = time.time()
        delaySleep = 0
        while not self._stopEv.wait(timeout=delaySleep):
            if self._enableCapture:
                ret, frame = self.read_frame()
                if ret:
                    self._frame = frame#cv2.resize(frame, (1920, 1080))

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            if delaySleep < 0:
                delaySleep = 0
                next_time = time.time()
        self.cleanup()

    def setExceptionMode(self, enable : bool):
        self.cap.setExceptionMode(enable)
    
    def open(self, src : typing.Union[int, str], fps : float = None) -> bool:
        ret = self.cap.open(src)
        if fps <= 0:
            fps = None
        if fps is None:
            fps = 30 #cap get fps
            #else: fps = 1000
        self._update_rate = fps
        return ret

    def start_capture(self):
        self._enableCapture = True

    def stop_capture(self):
        self._enableCapture = False
    
    def close(self):
        return self.cap.release()
        
    def read_frame(self) -> typing.Tuple[bool, np.ndarray]:
        return self.cap.read()
    
    def skip_frames(self, frames : int):
        for _ in range(frames):
            self.cap.grab()
    
    def isOpened(self) -> bool:
        return self.cap.isOpened()
    
    def frame(self) -> np.ndarray:
        return self._frame

class CameraSource(AIOTask, AsyncProcess):
    def __init__(self,
                 tm,
                 task_name,
                 src : typing.Union[int, str],
                 output : TaskOrTasks = None,
                 skip_frames : int = 0,
                 throttle_fps : float = None,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.source = src
        self._skipframes = skip_frames
        if self._skipframes < 0: self._skipframes = 0
        self.video_output = output
        self.throttle_fps = throttle_fps

        self.cap : CameraSourceProcess = None
        self._latest_frame : np.ndarray = None

        self._stop = asyncio.Event()
    
    async def start_task(self):
        await self.prepareProcess()
        self.cap = await self.asyncCreate(CameraSourceProcess)

    async def stop_task(self):
        self.logger.info("Stopping video capture...")
        await self.wait_task_timeout(5.0)

        if self.cap is not None:
            await self.cap.stop_capture()
            await self.cap.close()

        await self.endProcess()

    async def __call__(self):
        if self.cap is None:
            return
        
        self.logger.info("Starting video capture of source '%s'" % self.source)
        ret = await self.cap.open(self.source, self.throttle_fps)
        if not ret:
            self.logger.error("Capture could not be created at source '%s':" % self.source)
            
        await self.cap.skip_frames(self._skipframes)
        await self.cap.start_capture()
        
    def frame(self) -> asyncio.Future[np.ndarray]:
        return self.cap.frame()
