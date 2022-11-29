
import uuid
import typing
import asyncio
import logging

from concurrent.futures import ProcessPoolExecutor

class AsyncProcessMixin:
    '''
    '''
    def prepareProcess(self, workers : int = 1, **kwargs):
        self.proc : ProcessPoolExecutor = ProcessPoolExecutor(max_workers=workers, **kwargs)

    def endProcess(self):
        self.proc.shutdown()

    def processCall(self, func : typing.Callable, *args, logger : logging.Logger = None) -> asyncio.Future:
        task : asyncio.Future = asyncio.get_event_loop().run_in_executor(self.proc, func, *args)
        task.add_done_callback(self._proc_done_callback(logger))
        return task

    def _proc_done_callback(self, logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
        def _wraps(future : asyncio.Future):
            exc = future.exception()
            if exc:
                logger.exception("(Async process exception)", exc_info=exc)
        return _wraps

    
class AsyncProcess:
    '''
    '''
    def _mkinstance(cls, *args, **kwargs) -> str:
        all_items = globals()
        while True:
            i_id = str(uuid.uuid1())
            if i_id not in all_items: break
        all_items[i_id] = cls(*args, **kwargs)
        return i_id

    def _processCall(obj : str, name : str, *args, **kwargs):
        #Find instance and call the given method from that instance
        return getattr(globals()[obj], name)(*args, **kwargs)

    async def prepareProcess(self, **kwargs):
        self.proc : ProcessPoolExecutor = ProcessPoolExecutor(max_workers=1, **kwargs)

    async def endProcess(self):
        self.proc.shutdown()

    async def asyncCreate(self, cls : typing.Type, *args, **kwargs):
        '''
        Magic method to create new instance of a class, and manipulate the new instance remotely
        '''
        class _A(cls):
            def __init__(self_new, *passargs, **passkwargs):
                self._obj_id : str = self.proc.submit(AsyncProcess._mkinstance, cls, *passargs, **passkwargs).result()
            def __getattribute__(self_new, __name: str):
                #TODO: Attributes
                if callable(cls.__getattribute__(self_new, __name)):
                    def _remotemethod(*args, **kwargs) -> asyncio.Future:
                        #This should never happen
                        if not hasattr(self, '_obj_id'):
                            raise ValueError("Class %s must be instantiated using asyncCreate" % str(cls))
                        
                        task = asyncio.wrap_future(
                            self.proc.submit(AsyncProcess._processCall, self._obj_id, __name, *args, **kwargs)
                        )
                        task.add_done_callback(self._proc_done_callback())
                        return task
                    return _remotemethod
        return _A(*args, **kwargs)
    
    def _proc_done_callback(self, logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
        def _wraps(future : asyncio.Future):
            exc = future.exception()
            if exc:
                if isinstance(exc, KeyboardInterrupt):
                    logger.info("KeyboardInterrupt received")
                    return
                logger.exception("(Async process exception)", exc_info=exc)
        return _wraps

