
from vehiclebot.task import AIOTask, TaskOrTasks

from .camera import CameraSource
from vehiclebot.imutils import scaleImgRes
from vehiclebot.model.executor import ModelExecutor

import time
import asyncio
import functools

class RemoteObjectDetector(AIOTask):
    # Maximum requests/second to not overload
    ABSOLUTE_MAX_RPS = 100

    def __init__(self, tm, task_name,
                 input_source : str,
                 model : str,
                 output : TaskOrTasks = None,
                 process_size : int = None,
                 update_rate : float = None,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.inp_src = input_source
        self.output_result = output
        self.model_name = model
        self._process_size = process_size

        self._task_exec : ModelExecutor = self.tm.app['model_executor']

        self._update_rate = update_rate
        if self._update_rate is None:
            self._update_rate = 10

        self._stopEv = asyncio.Event()
        
    async def start_task(self):
        pass
            
    async def stop_task(self):
        self._stopEv.set()
        await self.wait_task_timeout()

    async def __call__(self):
        try:
            capTask : CameraSource = self.tm[self.inp_src]
        except KeyError:
            self.logger.error("Input source component \"%s\" is not loaded. Please check your config file to make sure it is properly configured." % self.inp_src)
            return

        next_time = time.time()
        delaySleep = 0
        while not await self._stopEv.wait_for(min(delaySleep, 1/self.ABSOLUTE_MAX_RPS)):
            #Try to grab a frame from video source
            img = capTask.frame()
            if img is not None:
                if self._process_size is not None:
                    img_proc, scale = await asyncio.get_event_loop().run_in_executor(None, functools.partial(scaleImgRes, img, height=self._process_size))
                else:
                    img_proc = img
                    scale = 1.0

                det_task = {
                    "task": "detect",
                    "params": {
                        "img": img_proc,
                        "model": self.model_name
                    }
                }
                det : dict = await self._task_exec.run(det_task)
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
