
import asyncio
import logging
import importlib
from aiohttp.web import Application

from concurrent.futures import Executor, ThreadPoolExecutor

from typing import Dict, Type, Any

def module_path_decode(module_string : str, default_package : str = __name__):
    pkg_split = module_string.split('.')
    return ('.'.join(pkg_split[:-1]) if len(pkg_split[:-1]) > 0 else default_package, pkg_split[-1])

class AIOTask(object):
    '''
    Base AIO task class
    '''
    metadata : Dict[str, Any] = {"dependencies": []}
    def __init__(self, tm, task_name : str, *args, **kwargs):
        self.tm = tm
        self.task : asyncio.Task = None
        self.name = task_name
        self.logger = logging.getLogger('>'.join([self.__module__, self.__class__.__qualname__, "$"+self.name]))
    async def start_task(self):
        pass
    async def stop_task(self):
        pass
    async def __call__(self):
        '''
        You can do whatever, as long as it is non-blocking and also somehow
        passes control back to asyncio (eg. using sleep, aio socket, etc.)
        '''
        pass

class TaskManager(object):
    def __init__(self, app : Application):
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.tasks : Dict[str, AIOTask] = dict()
        self.pool : Executor = ThreadPoolExecutor(max_workers=16)
        self.logger.info("Started ThreadPoolExecutor with %d threads" % self.pool._max_workers)
        
    async def add_task(self, task_name : str, task_conf : Dict):
        '''
        Add new task from configuration dict
        '''
        self.logger.info("Creating task \"%s\"" % task_name)
        if task_name in self.tasks:
            is_loaded = self.tasks[task_name].task is not None
            self.logger.warning("Task \"%s\ already exists %s, ignoring." % (task_name, "and is loaded" if is_loaded else ''))
            return

        task_cls : Type[AIOTask] = None
        
        imp_mod, imp_cls_name = module_path_decode(task_conf['type'], default_package='.'.join((__package__, 'components')))
        try:
            task_module = importlib.import_module(imp_mod)
            task_cls = getattr(task_module, imp_cls_name)
        except (ImportError, AttributeError):
            self.logger.error("Unable to load task plugin \"%s\" from module \"%s\". Make sure the module exists. Skipping" % (imp_cls_name, imp_mod))
            return
        if not issubclass(task_cls, AIOTask):
            self.logger.error("Cannot load task \"%s\". It must be of type AIOTask. Skipping" % imp_cls_name)
            return
        
        self.tasks[task_name] = task_cls(self, task_name, **task_conf.get('properties', {}))
        await self.tasks[task_name].start_task()
        self.tasks[task_name].task = asyncio.create_task(self.tasks[task_name]())
        return self.tasks[task_name]
        
    async def shutdown_task(self, task_name : str):
        self.logger.info("Shutting down task \"%s\"..." % task_name)
        await self.tasks[task_name].stop_task()
        
    async def enumerate_tasks(self, task_list : Dict[str, Dict]):
        await asyncio.gather(*[self.add_task(*k) for k in task_list.items()])

    async def close(self):
        await asyncio.gather(*[self.shutdown_task(k) for k in self.tasks.keys()])
        self.pool.shutdown()

    def __getitem__(self, idx : str):
        return self.tasks[idx]
