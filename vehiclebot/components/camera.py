
from vehiclebot.task import AIOTask, TaskOrTasks
from vehiclebot.imutils import scaleImgRes
from vehiclebot.model.filter import filter_pipeline

import time
import asyncio
import threading
import typing

import cv2
import numpy as np

class CameraSourceProcess(threading.Thread):
    def __init__(self, exception_mode : bool = False, preprocessor : typing.Callable = None):
        super().__init__(daemon=True)
        self._stopEv = threading.Event()
        self._frame : np.ndarray = None
        self._enableCapture = False
        self._update_rate : float = 1000
        self._callbacks = []
        self._enablePreprocess = True
        self._preprocess = preprocessor
        if self._preprocess is None:
            self._preprocess = lambda x: x

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
                    if self._enablePreprocess:
                        frame = self._preprocess(frame)
                    self._frame = frame
                    for cb in self._callbacks:
                        cb(self._frame)
                    
                    '''h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (int(w/2),int(h/2)))
                    pT = "filter: %s" % ('on' if self._enablePreprocess else 'off')
                    cv2.putText(frame, pT, (16,16), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), 3)
                    cv2.putText(frame, pT, (16,16), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)
                    cv2.imshow("full_frame", frame)
                    k = cv2.waitKey(1)
                    if k == ord('c'):
                        self._enablePreprocess = not self._enablePreprocess'''

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            if delaySleep < 0:
                delaySleep = 0
                #next_time = time.time()
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
    
    def putCallback(self, cb : typing.Callable):
        self._callbacks.append(cb)

class CameraSource(AIOTask):
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

        self.cap = None
        self._latest_frame : np.ndarray = None

        self._stopEv = asyncio.Event()
        self._frame_ready = asyncio.Event()
    
    def _cb_frame_ready(self, frame):
        self._latest_frame = frame
        asyncio.run_coroutine_threadsafe(self._set_frame_ready(), self.tm.loop)

    async def _set_frame_ready(self):
        self._frame_ready.set()

    async def start_task(self):
        self.cap = CameraSourceProcess(preprocessor=filter_pipeline)
        self.cap.putCallback(self._cb_frame_ready)

    async def stop_task(self):
        self.logger.info("Stopping video capture...")
        self._stopEv.set()
        self._frame_ready.set() #Notify

        if self.cap is not None:
            self.cap.stop_capture()
            await asyncio.get_event_loop().run_in_executor(None, self.cap.stop, 2)

        await self.wait_task_timeout(timeout=5.0)

    async def __call__(self):
        if self.cap is None:
            return
        
        self.logger.info("Starting video capture of source '%s'" % self.source)
        ret = await asyncio.get_event_loop().run_in_executor(None, self.cap.open, self.source, self.throttle_fps)
        if not ret:
            self.logger.error("Capture could not be created at source '%s':" % self.source)

        #Skip frames (optionally)
        await asyncio.get_event_loop().run_in_executor(None, self.cap.skip_frames, self._skipframes)

        await asyncio.sleep(5)
        self.cap.start_capture()

        while True:
            is_frame_ready = await self._frame_ready.wait_for(timeout=0.1)
            if is_frame_ready:
                self.emit("frame", self._latest_frame)
                self._frame_ready.clear()
            if self._stopEv.is_set():
                break
        
    def frame(self) -> np.ndarray:
        return self.cap.frame()
