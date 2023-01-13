
from ._base import ModelExecutor

import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from aiohttp.client import ClientSession
import aiohttp.client_exceptions
import mimetypes

import cv2
import numpy as np

import typing
from vehiclebot.types import Detections

#TODO: Extract method of the http request

class RemoteModelExecutor(ModelExecutor):
    TASK_EXEC_TABLE = {}

    def __init__(self, server_endpoint : str = None):
        self.logger = logging.getLogger(__name__)
        if server_endpoint is None:
            #TODO: Auto detect from app
            server_endpoint = "http://localhost:8080/"

        self.sess = ClientSession(base_url=server_endpoint)
        self.img_encode_format = '.png'

        # Thread pool to process synchronous code
        self._executor = ThreadPoolExecutor()

        #For delay after exception to not spam the server
        self._can_detect_after = time.time()
        self._can_ocr_after = time.time()

        self.TASK_EXEC_TABLE.update({
            'detect': self._detect,
            'ocr': self._ocr
        })

    def run(self, task : dict) -> asyncio.Task:
        '''
        Create asyncio Task from a dictionary of task name and parameters
        '''
        task_func_name = task["task"]
        task_func = self.TASK_EXEC_TABLE.get(task_func_name)
        if task_func is None:
            raise ValueError("Unknown function '%s' requested" % task_func_name)
        task_args = task.get('params') or {}
        return asyncio.create_task(task_func(**task_args))

    #### Functions to perform model actions

    async def _enc_img(self, img : np.ndarray, encode_format : str = ".png") -> typing.Tuple[bool, np.ndarray]:
        return await asyncio.get_event_loop().run_in_executor(self._executor, cv2.imencode, encode_format, img)

    async def _detect(self, img : np.ndarray, model : str = None):
        time_waiting = self._can_detect_after - time.time()
        if time_waiting > 0:
            await asyncio.sleep(time_waiting)

        encode_mime = mimetypes.types_map.get(self.img_encode_format)
        ret, img_blob = await self._enc_img(img, self.img_encode_format)
        if not ret:
            return

        #Send image to the model
        try:
            params = {}
            if model is not None:
                params.update({"model": model})
            response = await self.sess.post(
                '/detect',
                data=img_blob.tobytes(),
                params=params,
                headers={"Content-Type": encode_mime}
            )
            data = await response.json()
            return Detections(
                bboxes = np.array(data['detection']['bboxes']),
                detection_scores = np.array(data['detection']['detection_scores']),
                class_ids = np.array(data['detection']['class_ids'])
            )
        except aiohttp.client_exceptions.ClientConnectorError:
            retry_in_sec = 15.0
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


    async def _ocr(self, img : np.ndarray):
        time_waiting = time.time() - self._can_ocr_after
        if time_waiting > 0:
            await asyncio.sleep(time_waiting)

        encode_mime = mimetypes.types_map.get(self.img_encode_format)
        ret, img_blob = await self._enc_img(img, self.img_encode_format)

        if not ret:
            return

        try:
            sleepDelta = self._can_ocr_after - time.time()
            if sleepDelta > 0:
                self.logger.info("Waiting for %.2f seconds to try sending request...", sleepDelta)
                await asyncio.sleep(sleepDelta)

            response = await self.sess.post(
                '/recognize',
                data=img_blob.tobytes(),
                params={},
                headers={"Content-Type": encode_mime}
            )
            return await response.json()
        except aiohttp.client_exceptions.ClientConnectorError:
            #Connection straight up failed
            retry_in_sec = 15.0
            self.logger.warning("Remote detection server (%s) is unreachable. Will retry after %.1f seconds" % (self.sess._base_url, retry_in_sec))
            self._can_ocr_after = time.time() + retry_in_sec
        except aiohttp.client_exceptions.ContentTypeError:
            #Data not in expected JSON format
            retry_in_sec = 1.0
            self.logger.exception("Incorrect response, retrying after %.1f seconds:" % retry_in_sec)
            self.logger.info("Content of above exception:\n<<<<<<<\n%s\n>>>>>>>" % await response.text())
            self._can_ocr_after = time.time() + retry_in_sec
        except asyncio.CancelledError:
            #Task cancelled
            pass
        except:
            #Some other error
            retry_in_sec = 1.0
            self.logger.exception("Remote detection error, retrying after %.1f seconds:" % retry_in_sec)
            self._can_ocr_after = time.time() + retry_in_sec
