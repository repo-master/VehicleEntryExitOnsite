
from vehiclebot.task import AIOTask
from vehiclebot.management.rtc import DataChannelHandler

import typing

class RTCDataProcess(AIOTask):
    def __init__(self, *args, plate_decoder = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.handlers : typing.Dict[str, DataChannelHandler] = dict(
            cmd = DataChannelHandler(),
            log = DataChannelHandler(),
            status = DataChannelHandler()
        )

    async def start_task(self):
        await super().start_task()
        self.tm.app['rtc'].addInDataChannelHandler(**self.handlers)
        self.on('update', self.broadcast_update_data)

    async def broadcast_update_data(self, vehicles):
        pass