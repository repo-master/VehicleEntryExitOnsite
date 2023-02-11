
from vehiclebot.task import AIOTask
import typing
from contextlib import suppress

class Link(AIOTask):
    '''
    Join a task emitting event to self with a task (or tasks) accepting events
    by forwarding that event
    '''
    def __init__(self, tm, task_name : str,
                task_input : typing.Dict[str, str],
                task_output : typing.Dict[str, str] = {}):
        super().__init__(tm, task_name)
        self._in = task_input
        self._out = task_output

    def makeEVProxy(self, task : AIOTask, event : str):
        async def _proxy(*args, **kwargs):
            task.emit(event, *args, **kwargs)
        return _proxy

    async def __call__(self):
        for it, evin in self._in.items():
            it_task = self.tm[it]
            for ot, evout in self._out.items():
                ot_task = self.tm[ot]
                _proxy = self.makeEVProxy(ot_task, evout)
                it_task.add_listener(evin, _proxy)
