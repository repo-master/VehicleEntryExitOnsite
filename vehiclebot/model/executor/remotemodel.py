
from ._base import ModelExecutor

import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import requests
from urllib.parse import urljoin

from aiohttp.client import ClientSession
import aiohttp.client_exceptions
import mimetypes

import cv2
import numpy as np

import typing
from vehiclebot.types import Detections

#TODO: Extract the program code of the http request to another method

class RemoteModelExecutor(ModelExecutor):
    TASK_EXEC_TABLE = {}

    def __init__(self, server_endpoint : str = None, img_encode_format : str = ".png"):
        self.logger = logging.getLogger(__name__)
        if server_endpoint is None:
            #TODO: Auto detect from app
            server_endpoint = "http://localhost:8080/"

        self.sess = ClientSession(base_url=server_endpoint)
        self.img_encode_format = img_encode_format

        # Thread pool to process synchronous code
        self._executor = ThreadPoolExecutor()

        #For delay after exception to not spam the server
        self._can_detect_after = time.time()
        self._can_ocr_after = time.time()

        self.TASK_EXEC_TABLE.update({
            'detect': self._detect,
            'ocr': self._ocr
        })

        self.logger.info("Started %s connecting to endpoint '%s'", self.__class__.__name__, server_endpoint)

    async def _cleanup(self):
        await self.sess.close()

    def run(self, task_name : str, *args, **kwargs) -> asyncio.Task:
        '''
        Create asyncio Task from the given task name and parameters
        '''
        task_func = self.TASK_EXEC_TABLE.get(task_name)
        if task_func is None:
            raise ValueError("Unknown function '%s' requested" % task_name)
        return asyncio.create_task(task_func(*args, **kwargs))

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
        except KeyboardInterrupt:
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
        except KeyboardInterrupt:
            pass
        except:
            #Some other error
            retry_in_sec = 1.0
            self.logger.exception("Remote detection error, retrying after %.1f seconds:" % retry_in_sec)
            self._can_ocr_after = time.time() + retry_in_sec

class RemoteThreadedModelExecutor(ModelExecutor):
    TASK_EXEC_TABLE = {}

    def __init__(self, server_endpoint : str = None, img_encode_format : str = ".png"):
        self.logger = logging.getLogger(__name__)
        if server_endpoint is None:
            #TODO: Auto detect from app
            server_endpoint = "http://localhost:8080/"

        self.server_endpoint = server_endpoint
        self.img_encode_format = img_encode_format

        #Thread pool for requests
        self._executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="RTME")

        #For delay after exception to not spam the server
        self._can_detect_after = time.time()
        self._can_ocr_after = time.time()

        self.TASK_EXEC_TABLE.update({
            'detect': self._detect,
            'ocr': self._ocr
        })

        self.logger.info("Started %s connecting to endpoint '%s'", self.__class__.__name__, server_endpoint)


    def run(self, task_name : str, *args, **kwargs) -> asyncio.Task:
        '''
        Create asyncio Task from the given task name and parameters
        '''
        task_func = self.TASK_EXEC_TABLE.get(task_name)
        if task_func is None:
            raise ValueError("Unknown function '%s' requested" % task_name)
        return asyncio.get_event_loop().run_in_executor(self._executor, functools.partial(task_func, *args, **kwargs))

    #### Functions to perform model actions

    def _enc_img(self, img : np.ndarray, encode_format : str = ".png") -> typing.Tuple[bool, np.ndarray]:
        return cv2.imencode(encode_format, img)

    def _detect(self, img : np.ndarray, model : str = None, **kwargs):
        time_waiting = self._can_detect_after - time.time()
        if time_waiting > 0:
            time.sleep(time_waiting)

        encode_mime = mimetypes.types_map.get(self.img_encode_format)
        ret, img_blob = self._enc_img(img, self.img_encode_format)
        if not ret:
            return

        #Send image to the model
        try:
            params = {}
            if model is not None:
                params.update({"model": model})
            params.update(kwargs)
            response = requests.post(
                urljoin(self.server_endpoint, '/detect'),
                data=img_blob.tobytes(),
                params=params,
                headers={"Content-Type": encode_mime}
            )
            data = response.json()
            return Detections(
                bboxes = np.array(data['detection']['bboxes']),
                detection_scores = np.array(data['detection']['detection_scores']),
                class_ids = np.array(data['detection']['class_ids'])
            )
        except requests.ConnectionError:
            retry_in_sec = 5.0
            self.logger.warning("Remote detection server (%s) is unreachable. Will retry after %.1f seconds" % (self.server_endpoint, retry_in_sec))
            self._can_detect_after = time.time() + retry_in_sec
        except requests.JSONDecodeError:
            retry_in_sec = 1.0
            self.logger.exception("Incorrect detection response, retrying after %.1f seconds:" % retry_in_sec)
            self.logger.info("Content of above exception:\n<<<<<<<\n%s\n>>>>>>>" % response.text)
            self._can_detect_after = time.time() + retry_in_sec
        except KeyboardInterrupt:
            pass
        except:
            retry_in_sec = 1.0
            self.logger.exception("Remote detection error, retrying after %.1f seconds:" % retry_in_sec)
            self._can_detect_after = time.time() + retry_in_sec


    def _ocr(self, img : np.ndarray):
        sleepDelta = self._can_ocr_after - time.time()
        if sleepDelta > 0:
            self.logger.info("Waiting for %.2f seconds to try sending request...", sleepDelta)
            time.sleep(sleepDelta)

        encode_mime = mimetypes.types_map.get(self.img_encode_format)
        ret, img_blob = self._enc_img(img, self.img_encode_format)

        if not ret:
            return

        try:
            response = requests.post(
                urljoin(self.server_endpoint, '/recognize'),
                data=img_blob.tobytes(),
                params={},
                headers={"Content-Type": encode_mime}
            )
            return response.json()
        except requests.ConnectionError:
            #Connection straight up failed
            retry_in_sec = 5.0
            self.logger.warning("Remote detection server (%s) is unreachable. Will retry after %.1f seconds" % (self.sess._base_url, retry_in_sec))
            self._can_ocr_after = time.time() + retry_in_sec
        except requests.JSONDecodeError:
            #Data not in expected JSON format
            retry_in_sec = 1.0
            self.logger.exception("Incorrect response, retrying after %.1f seconds:" % retry_in_sec)
            self.logger.info("Content of above exception:\n<<<<<<<\n%s\n>>>>>>>" % response.text)
            self._can_ocr_after = time.time() + retry_in_sec
        except KeyboardInterrupt:
            pass
        except:
            #Some other error
            retry_in_sec = 1.0
            self.logger.exception("Remote detection error, retrying after %.1f seconds:" % retry_in_sec)
            self._can_ocr_after = time.time() + retry_in_sec
