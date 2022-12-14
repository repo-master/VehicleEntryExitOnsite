
import asyncio
import logging
import importlib

from aiohttp.web import Application
from .task import AIOTask

from concurrent.futures import Executor, ThreadPoolExecutor

import tqdm.asyncio
from tqdm.contrib.logging import logging_redirect_tqdm

from typing import Dict, List, Type, Union

def module_path_decode(module_string : str, default_package : str = __name__):
    pkg_split = module_string.split('.')
    return ('.'.join(pkg_split[:-1]) if len(pkg_split[:-1]) > 0 else default_package, pkg_split[-1])

class TaskManager(object):
    def __init__(self, app : Application):
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.tasks : Dict[str, AIOTask] = dict()
        self.pool : Executor = ThreadPoolExecutor(max_workers=4)
        self.logger.info("Started ThreadPoolExecutor with %d threads" % self.pool._max_workers)
        
    def _handle_task_done(self, cor : asyncio.Task):
        res = cor.result()
        if res is not None:
            self.logger.info("Task %s completed with result: %s", cor.get_name(), res)
        else:
            self.logger.debug("Task %s closed", cor.get_name())

    async def add_task(self, task_name : str, task_conf : Dict):
        '''
        Add new task from configuration dict
        '''
        self.logger.info("Creating task \"%s\"..." % task_name)
        if task_name in self.tasks:
            is_loaded = self.tasks[task_name].task is not None
            self.logger.warning("Task \"%s\ already exists %s, ignoring." % (task_name, "and is loaded" if is_loaded else ''))
            return

        task_cls : Type[AIOTask] = None
        
        imp_mod, imp_cls_name = module_path_decode(task_conf['type'], default_package='.'.join((__package__, 'components')))
        try:
            self.logger.debug("Importing module '%s' for '%s'" % (imp_mod, task_name))
            task_module = importlib.import_module(imp_mod)
            task_cls = getattr(task_module, imp_cls_name)
        except (ImportError, AttributeError):
            self.logger.exception("Unable to load task plugin \"%s\" from module \"%s\". Make sure the module exists. Skipping" % (imp_cls_name, imp_mod))
            return
        if not issubclass(task_cls, AIOTask):
            self.logger.error("Cannot load task \"%s\". It must be of type AIOTask. Skipping" % imp_cls_name)
            return
        
        #Instantiate class with parameters
        props = task_conf.get('properties') or {}
        self.tasks[task_name] = task_cls(self, task_name, **props)
        await self.tasks[task_name].start_task()
        self.tasks[task_name].task = asyncio.create_task(self.tasks[task_name]())
        self.tasks[task_name].task.add_done_callback(self._handle_task_done)
        self.logger.info("Task \"%s\" (id: %s) created" % (task_name, self.tasks[task_name].task.get_name()))
        return self.tasks[task_name]
        
    async def shutdown_task(self, task_name : str):
        self.logger.info("Shutting down task \"%s\"..." % task_name)
        await self.tasks[task_name].stop_task()
        
    async def enumerate_tasks(self, task_list : Dict[str, Dict]):
        task_list = [self.add_task(*k) for k in task_list.items()]
        with logging_redirect_tqdm():
            return [await f for f in tqdm.asyncio.tqdm.as_completed(task_list)]
            
    async def close(self):
        task_list = [self.shutdown_task(k) for k in self.tasks.keys()]
        with logging_redirect_tqdm():
            for f in tqdm.asyncio.tqdm.as_completed(task_list):
                await f
        self.pool.shutdown()

    async def emit(self, task : Union[AIOTask, str, List[Union[AIOTask, str]]], event : str, *args, **kwargs) -> Union[bool, List[bool]]:
        if task is None: return False
        if isinstance(task, list):
            return await asyncio.gather(*[self.emitTask(x, event, *args, **kwargs) for x in task])
        task_obj : AIOTask = task
        if isinstance(task, str):
            try:
                task_obj = self[task]
            except KeyError:
                self.logger.error("Task '%s' not found", task)
                return False
        return task_obj.emit(event, *args, **kwargs)

    def __getitem__(self, idx : str):
        return self.tasks[idx]
