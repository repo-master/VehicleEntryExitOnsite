
from vehiclebot.task import AIOTask
from vehiclebot.types import PlateDetection

import cv2
import asyncio
import numpy as np

import time
import mimetypes
import aiohttp.client_exceptions

import typing

class PlateRecognizer(AIOTask):
    metadata : typing.Dict[str, typing.Any] = {"dependencies": []}
    def __init__(self, tm, task_name,
                 img_format : str = ".png",
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.sess = self.tm.app.cli
        self.img_encode_format = img_format
        self._can_detect_after = time.time()
        
    async def start_task(self):
       self.on('recognize', self.detectAndDecode)
        
    async def stop_task(self):
        await self.task
        
    async def detectAndDecode(self, detection : dict) -> PlateDetection:
        #TODO: Choose which images to send by measuring likelyhood of getting detection
        img = detection.get('img')
        if img is None: return
        encode_mime = mimetypes.types_map.get(self.img_encode_format)
        ret, img_blob = cv2.imencode(self.img_encode_format, img)
        if not ret: return
        #Send image to the model
        try:
            sleepDelta = self._can_detect_after - time.time()
            if sleepDelta > 0:
                self.logger.info("Waiting for %.2f seconds to try sending request...", sleepDelta)
                await asyncio.sleep(sleepDelta)

            response = await self.sess.post(
                "/recognize",
                data=img_blob.tobytes(),
                params={},
                headers={"Content-Type": encode_mime}
            )
            return await response.json()
        except aiohttp.client_exceptions.ClientConnectorError:
            retry_in_sec = 20.0
            self.logger.warning("Remote detection server (%s) is unreachable. Will retry after %.1f seconds" % (self.sess._base_url, retry_in_sec))
            self._can_detect_after = time.time() + retry_in_sec
        except aiohttp.client_exceptions.ContentTypeError:
            retry_in_sec = 1.0
            self.logger.exception("Incorrect plate detection response, retrying after %.1f seconds:" % retry_in_sec)
            self.logger.info("Content of above exception:\n>==========>\n%s\n<==========<" % await response.text())
            self._can_detect_after = time.time() + retry_in_sec
        except:
            retry_in_sec = 1.0
            self.logger.exception("Remote detection error, retrying after %.1f seconds:" % retry_in_sec)
            self._can_detect_after = time.time() + retry_in_sec

