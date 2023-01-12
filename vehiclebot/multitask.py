
import uuid
import typing
import signal
import asyncio
import logging
import threading

from concurrent.futures import Future, ProcessPoolExecutor

# def wrapAsyncThreadExecutor(cls : typing.Type[threading.Thread]):
#     '''
#     Makes any class function run 
#     '''
#     class _A(cls):
#         def __getattribute__(self, __name: str):
#             _orig_attr = cls.__getattribute__(self, __name)
#             if callable(_orig_attr):
#                 def _futuremethod(*args, **kwargs) -> asyncio.Future:
#                     _f = Future()
#                     task = asyncio.wrap_future(_f)
#                     task.add_done_callback(self._proc_done_callback())
#                     return task
#                 return _futuremethod
#             return _orig_attr
#         def _proc_done_callback(self, logger=None):
#             if logger is None:
#                 logger = logging.getLogger(__name__)
#             def _wraps(future : asyncio.Future):
#                 exc = future.exception()
#                 if exc:
#                     if isinstance(exc, KeyboardInterrupt):
#                         logger.info("KeyboardInterrupt received")
#                         return
#                     logger.exception("(Async executor exception)", exc_info=exc)
#             return _wraps
#     return _A

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
                logger.exception("(Async executor exception)", exc_info=exc)
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

    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _processCall(obj : str, name : str, *args, **kwargs):
        #Find instance and call the given method from that instance
        return getattr(globals()[obj], name)(*args, **kwargs)

    async def prepareProcess(self, **kwargs):
        self.proc : ProcessPoolExecutor = ProcessPoolExecutor(max_workers=1, initializer=AsyncProcess.init_worker, **kwargs)

    async def endProcess(self):
        self.proc.shutdown()

    async def asyncCreate(self, cls : typing.Type, *args, **kwargs):
        '''
        Magic method to create new instance of a class, and manipulate the new instance remotely
        '''
        class _A(cls):
            def __init__(self_new, *passargs, **passkwargs):
                _obj_make = self.proc.submit(AsyncProcess._mkinstance, cls, *passargs, **passkwargs)
                _obj_make.add_done_callback(self._proc_done_callback())
                self._obj_id : str = _obj_make.result()
            def __getattribute__(self_new, __name: str):
                _orig_attr = cls.__getattribute__(self_new, __name)
                if callable(_orig_attr) and not asyncio.iscoroutine(_orig_attr):
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
                return _orig_attr
        return _A(*args, **kwargs)
    
    def _proc_done_callback(self, logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
        def _wraps(future : asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    if isinstance(exc, KeyboardInterrupt):
                        logger.info("KeyboardInterrupt received")
                        return
                    logger.exception("(Async executor exception)", exc_info=exc)
            except asyncio.CancelledError as ex:
                logger.warning("A future was cancelled before completing: %s" % future, exc_info=ex)
            except asyncio.InvalidStateError as ex:
                logger.debug("Future was still pending: %s" % future, exc_info=ex)
        return _wraps

