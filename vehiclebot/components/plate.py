
from vehiclebot.task import AIOTask
from vehiclebot.types import PlateDetection

import cv2
import time
import asyncio
import numpy as np

import mimetypes
from aiohttp.client import ClientSession
import aiohttp.client_exceptions

import typing

class PlateRecognizer(AIOTask):
    metadata : typing.Dict[str, typing.Any] = {"dependencies": []}
    def __init__(self, tm, task_name,
                 img_format : str = ".png",
                 server_endpoint : str = None,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)

        self._plate_endpoint = "/recognize"

        if server_endpoint is None:
            #TODO: Auto detect
            server_endpoint = "http://localhost:8080/"
            self._plate_endpoint = "/model" + self._plate_endpoint
            if 'server' not in self.tm.app:
                self.logger.warning("Server endpoint for model server not defined, and integrated server is not active. Detection may fail.")
        
        self.sess = ClientSession(base_url=server_endpoint)

        self.img_encode_format = img_format
        self._can_detect_after = time.time()
        
    async def start_task(self):
       self.on('recognize', self.detectAndDecode)
        
    async def stop_task(self):
        await self.wait_task_timeout()
        
    async def detectAndDecode(self, detection : dict, when : float = None) -> PlateDetection:
        if when is None: when = time.time()
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

            wait_delay = max(0, when - time.time())
            if wait_delay > 0:
                await asyncio.sleep(wait_delay)

            response = await self.sess.post(
                self._plate_endpoint,
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
            self.logger.info("Content of above exception:\n<<<<<<<\n%s\n>>>>>>>" % await response.text())
            self._can_detect_after = time.time() + retry_in_sec
        except asyncio.CancelledError:
            #Task cancelled
            pass
        except:
            retry_in_sec = 1.0
            self.logger.exception("Remote detection error, retrying after %.1f seconds:" % retry_in_sec)
            self._can_detect_after = time.time() + retry_in_sec

