
from vehiclebot.task import AIOTask
from vehiclebot.multitask import AsyncProcessMixin

import asyncio
import cv2
from decouple import config

HEADLESS = config('HEADLESS', cast=bool, default=False)

class VideoDisplayProcess:
    @staticmethod
    def makeDPIAware():
        #Windows's scaling issue
        import ctypes
        if hasattr(ctypes, 'windll'):
            ctypes.windll.user32.SetProcessDPIAware()
        
    @staticmethod
    def imshow(window_name, img):
        if HEADLESS: return
        global writers
        try:
            writers
        except NameError:
            writers = dict()

        if window_name not in writers:
            writers[window_name] = cv2.VideoWriter(
                "results/%s.m4v" % window_name,
                cv2.VideoWriter_fourcc(*'MP4V'), 4, (1080, 1920)
            )
        
        writers[window_name].write(img)
        
        #img = cv2.resize(img, (360, 640))
        cv2.imshow(window_name, img)
    
    @staticmethod
    def destroyAllWindows():
        for x, y in writers.items():
            y.release()
        cv2.destroyAllWindows()
    
    @staticmethod
    def waitKey(delay):
        return cv2.waitKey(delay)

class VideoDisplay(AIOTask, AsyncProcessMixin):
    def __init__(self, tm, task_name, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self._stop = asyncio.Event()
        self.prepareProcess(initializer=VideoDisplayProcess.makeDPIAware)
    
    async def stop_task(self):
        self.logger.info("Stopping video display")
        self._stop.set()
        await self.task
        self.endProcess()

    async def __call__(self):
        while not self._stop.is_set():
            #Poll window manager
            key = await self.processCall(VideoDisplayProcess.waitKey, 15)

            #if key == ord('q'):
            #    self._stop = True
            #    break
        await self.processCall(VideoDisplayProcess.destroyAllWindows)
    
    async def imshow(self, window_name, img):
        return await self.processCall(VideoDisplayProcess.imshow, window_name, img)

