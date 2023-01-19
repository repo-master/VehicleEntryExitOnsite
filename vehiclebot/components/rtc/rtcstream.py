
from vehiclebot.task import AIOTask

from aiortc import VideoStreamTrack
from av.frame import Frame
from av import VideoFrame

import cv2
import numpy as np

import typing

#TODO: Use relay to broadcast

class RTCStreamTrack(AIOTask, VideoStreamTrack):
    def __init__(self, tm, task_name, resolution : typing.Dict[str, int] = None, **kwargs):
        AIOTask.__init__(self, tm, task_name, **kwargs)
        VideoStreamTrack.__init__(self)
        init_resoltion = dict(width=640, height=360)

        self._scaleResolution = resolution
        if  self._scaleResolution is not None:
            init_resoltion = self._scaleResolution
        self._frame = VideoFrame(**init_resoltion)
        
    async def start_task(self):
        self.tm.app['rtc'].addMediaTrack(self.name, self)
        self.on('frame', self.imshow)

    async def stop_task(self):
        self.logger.debug("Ending video stream...")
        VideoStreamTrack.stop(self)
        self.logger.info("Video stream ended")
        await self.task
        
    async def imshow(self, window_name : str, img : np.ndarray):
        if img is None: return
        if self._scaleResolution is not None:
            img = cv2.resize(img, (self._scaleResolution['width'], self._scaleResolution['height']))
        self._frame = VideoFrame.from_ndarray(img, format='bgr24')


    ## Methods for aiortc

    # Override from VideoStreamTrack
    async def recv(self) -> Frame:
        pts, time_base = await self.next_timestamp()
        
        #Presentation timestamp
        self._frame.pts = pts
        self._frame.time_base = time_base
        return self._frame
    
    # Override from MediaStreamTrack
    def stop(self) -> None:
        #No need to stop track if connection closes
        pass
