
from vehiclebot.task import AIOTask
from vehiclebot.multitask import AsyncProcess

import asyncio
import typing
import cv2

#Synchronous code in isolated process
#This is needed because OpenCV function calls are not async
class CameraSourceProcess:
    def __init__(self):
        self.cap = cv2.VideoCapture()
        self.cap.setExceptionMode(True)

    def setExceptionMode(self, enable : bool):
        self.cap.setExceptionMode(enable)
    
    def start_capture(self, src : typing.Union[int, str]) -> bool:
        return self.cap.open(src)
    
    def stop_capture(self):
        self.cap.release()
        
    def read_frame(self):
        return self.cap.read()
    
    def skip_frames(self, frames : int):
        for _ in range(frames):
            self.cap.grab()
    
    def isOpened(self) -> bool:
        return self.cap.isOpened()

class CameraSource(AIOTask, AsyncProcess):
    def __init__(self, tm, task_name, src : typing.Union[int, str], skip_frames : int = 0, video_output : str = None, throttle_fps : float = None, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.source = src
        self._skipframes = skip_frames
        if self._skipframes < 0: self._skipframes = 0
        self.video_output = video_output
        self.throttle_fps = throttle_fps

        self.cap : CameraSourceProcess = None
        self._latest_frame = None
        self.cap_get_wait = asyncio.Event()

        self._stop = asyncio.Event()
        self.prepareProcess()
        self.logger.info("Started OpenCV worker process")
    
    async def start_task(self):
        self.logger.info("Starting video capture of source \"%s\"" % str(self.source))
        try:
            self.cap = await self.asyncCreate(CameraSourceProcess)
            await self.cap.start_capture(self.source)
        except Exception:
            self.logger.exception("Capture could not be created at source '%s':" % self.source)
        finally:
            self.cap_get_wait.set()

    async def stop_task(self):
        self._stop.set()
        self.logger.info("Stopping video capture")
        await self.task
        if self.cap is not None:
            try:
                await self.cap.stop_capture()
            except Exception:
                self.logger.exception("Failed to stop capture")

        self.endProcess()

    async def __call__(self):
        await self.cap_get_wait.wait()
        if self.cap is None: return

        #Skip frames
        try:
            await self.cap.skip_frames(self._skipframes)
        except Exception:
            self.logger.exception("Failed to skip frames")

        await self.cap.setExceptionMode(False)

        while not self._stop.is_set():
            #Fetch images as fast as possible
            ret, img = await self.cap.read_frame()
            if ret:
                self._latest_frame = img
                if self.video_output is not None:
                    await self.tm[self.video_output].imshow("Video", img)

                if self.throttle_fps is not None:
                    await asyncio.sleep(1.0 / self.throttle_fps)
            else:
                self.logger.info("Video capture has stopped")
                break

    @property
    def frame(self):
        return self._latest_frame
