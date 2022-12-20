
from vehiclebot.task import AIOTask, TaskOrTasks

from .camera import CameraSource
from vehiclebot.types import Detections
from vehiclebot.imutils import scaleImgRes

import cv2
import time
import asyncio
import numpy as np

import mimetypes
from aiohttp.client import ClientSession
import aiohttp.client_exceptions

class RemoteObjectDetector(AIOTask):
    def __init__(self, tm, task_name,
                 input_source : str,
                 model : str,
                 output : TaskOrTasks = None,
                 process_size : int = None,
                 img_format : str = ".png",
                 server_endpoint : str = None,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.inp_src = input_source
        self.output_result = output
        self.model_name = model
        self.img_encode_format = img_format
        self._process_size = process_size
        
        self._update_rate = 10
        self._can_detect_after = time.time()

        self._detect_endpoint = "/detect"

        if server_endpoint is None:
            #TODO: Auto detect
            server_endpoint = "http://localhost:8080/"
            self._detect_endpoint = "/model" + self._detect_endpoint
            if 'server' not in self.tm.app:
                self.logger.warning("Server endpoint for model server not defined, and integrated server is not active. Detection may fail.")
        
        self.sess = ClientSession(base_url=server_endpoint)
        
        self._stopEv = asyncio.Event()
        
    async def start_task(self):
        pass
            
    async def stop_task(self):
        self._stopEv.set()
        await self.wait_task_timeout()
        await self.sess.close()

    async def __call__(self):
        try:
            capTask : CameraSource = self.tm[self.inp_src]
        except KeyError:
            self.logger.error("Input source component \"%s\" is not loaded. Please check your config file to make sure it is properly configured." % self.inp_src)
            return
        
        next_time = time.time()
        delaySleep = 0
        while not await self._stopEv.wait_for(min(delaySleep, 1/100)): #Do not exceed 100 reqs/s
            if time.time() > self._can_detect_after:
                #Try to grab a frame from video source
                img = await capTask.frame()
                if img is not None:
                    if self._process_size is not None:
                        img_proc, scale = scaleImgRes(img, height=self._process_size)
                    else:
                        img_proc = img
                        scale = 1.0
                    det = await self.detect(img_proc)
                    if det:
                        await self.tm.emit(
                            self.output_result,
                            "detect",
                            detection=det,
                            frame=img,
                            scale=scale
                        )

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
                
    async def detect(self, img : np.ndarray) -> Detections:
        if img is None: return
        encode_mime = mimetypes.types_map.get(self.img_encode_format)
        ret, img_blob = cv2.imencode(self.img_encode_format, img)
        if not ret: return
        #Send image to the model
        try:
            response = await self.sess.post(
                self._detect_endpoint,
                data=img_blob.tobytes(),
                params={"model": self.model_name},
                headers={"Content-Type": encode_mime}
            )
            data = await response.json()
            return Detections(
                bboxes = np.array(data['detection']['bboxes']),
                detection_scores = np.array(data['detection']['detection_scores']),
                class_ids = np.array(data['detection']['class_ids'])
            )
        except aiohttp.client_exceptions.ClientConnectorError:
            retry_in_sec = 20.0
            self.logger.warning("Remote detection server (%s) is unreachable. Will retry after %.1f seconds" % (self.sess._base_url, retry_in_sec))
            self._can_detect_after = time.time() + retry_in_sec
        except aiohttp.client_exceptions.ContentTypeError:
            retry_in_sec = 1.0
            self.logger.exception("Incorrect detection response, retrying after %.1f seconds:" % retry_in_sec)
            self.logger.info("Content of above exception:\n<<<<<<<\n%s\n>>>>>>>" % await response.text())
            self._can_detect_after = time.time() + retry_in_sec
        except asyncio.CancelledError:
            #Task cancelled
            pass
        except:
            retry_in_sec = 1.0
            self.logger.exception("Remote detection error, retrying after %.1f seconds:" % retry_in_sec)
            self._can_detect_after = time.time() + retry_in_sec
