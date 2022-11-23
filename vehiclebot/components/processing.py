
import typing
from vehiclebot.task import AIOTask
from vehiclebot.patches import JSONifier

import os
import asyncio
import json
import base64

import numpy as np

class DetectionHandler(AIOTask):
    '''
    Send detection data to a remote service for further processing.
    This is the base class with stub method _dispatchDetection().
    '''

    def __init__(self, tm, task_name, max_queue : int = 500, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.save_queue = asyncio.Queue(maxsize=max_queue)
        self._stop = asyncio.Event()

    async def stop_task(self):
        self._stop.set()
        await self.task

    async def __call__(self):
        while not self._stop.is_set():
            is_stop, next_items = await asyncio.gather(self._stop.wait_for(timeout=.1), self.save_queue.get_wait_for(timeout=.3))
            if is_stop: break
            if next_items is None: continue
            self.tm.pool.map(self._dispatchDetection, next_items)
            self.save_queue.task_done()

    async def processDetection(self, detection_task : asyncio.Future):
        # If queue gets full, detection process will get paused due to queue blocking
        await self.save_queue.put(detection_task)
        
    def _dispatchDetection(self, detection : dict):
        pass

class DirectoryProcess(DetectionHandler):
    '''
    Handles detected result by saving it into a directory on the disk.
    Image file and metadata are saved separately with same file name,
    or can be saved in one file with the image as a blob
    '''

    DEFAULT_FNAME_FMT = "{first_detect_ts_fmt}_{class}_{track_id}-{age}"

    def __init__(self, *args, dir : os.PathLike = "results/", save_mode : str = "file", filename_format : str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dest_dir = dir
        self.save_mode = save_mode
        self.fname_fmt = filename_format
        if self.fname_fmt is None:
            self.fname_fmt = self.DEFAULT_FNAME_FMT
        
    async def start_task(self):
        if not os.path.isdir(self.dest_dir):
            self.logger.warning("The path \"%s\" does not exist. Creating..." % self.dest_dir)
            os.makedirs(self.dest_dir)
        return await super().start_task()
    
    def _saveMetadataObject(self, detection : dict, save_meta_file : os.PathLike):
        if detection['is_new_detection']:
            self.logger.debug("New detection: Saving detection to \"%s\"" % save_meta_file)
        with open(save_meta_file, 'w') as meta_file:
            json.dump(detection, meta_file, cls=JSONifier, default=str)

    def _saveImageObject(self, img_blob : bytes, save_img_file : os.PathLike):
        with open(save_img_file, 'wb') as img_file:
            img_file.write(img_blob)
            
    def _dispatchDetection(self, detection : dict):
        img_data : dict = detection.pop('img')
        if img_data is None: return
        img_blob = None

        if img_data['encoding'] == 'encoded_np':
            img_bytes : np.ndarray = img_data['data']
            assert isinstance(img_bytes, np.ndarray)
            img_blob = img_bytes.tobytes()
        else:
            img_blob = img_data['data']

        #Extra formatting data for file name
        fmt_data = {
            'first_detect_ts_fmt': detection['first_detect_ts'].strftime("%Y-%m-%d_%H-%M")
        }
        save_fname = self.fname_fmt.format(**detection, **fmt_data)
        save_fullpath = os.path.join(self.dest_dir, save_fname)
        
        save_meta_file = save_fullpath + '.json'
        
        if not isinstance(img_blob, bytes):
            self.logger.warning("Detection \"%s\" has invalid image data encoded as \"%s\". Image may be corrupted." % (save_meta_file, img_data['encoding']))
            
        ## Change format to one suitable for storage
        if self.save_mode == 'file':
            #Save image as separate file
            save_img_file = save_fullpath + img_data['format']
            detection['img'] = {
                'encoding': self.save_mode,
                'format': img_data['format'],
                'file': os.path.relpath(save_img_file, self.dest_dir)
            }
            
            #Save the image
            self._saveImageObject(img_blob, save_img_file)
            
            #Save metadata file
            self._saveMetadataObject(detection, save_meta_file)
        elif self.save_mode == 'embedded':
            #Save image within the metadata json file
            if img_blob is None:
                self.logger.warning("Detection \"%s\" has invalid image data encoded as \"%s\". Image not written" % (save_meta_file, img_data['encoding']))
            
            detection['img'] = {
                'encoding': self.save_mode,
                'format': img_data['format'],
                'blob': base64.b64encode(img_blob).decode()
            }
            
            #Save metadata file
            self._saveMetadataObject(detection, save_meta_file)
        else:
            raise ValueError("Invalid save mode passed \"%s\"" % self.save_mode)
            
class RemoteProcess(DetectionHandler):
    pass
