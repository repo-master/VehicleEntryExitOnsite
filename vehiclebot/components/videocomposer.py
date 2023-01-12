
import typing
from vehiclebot.task import AIOTask

class VehicleManager(AIOTask):
    def __init__(self, tm, task_name,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.videoLayers : typing.Dict[int, str] = dict()

    async def __call__(self):
        pass

    async def setVideoLayerInputSource(self, layer : int, listen_event : str):
        pass
