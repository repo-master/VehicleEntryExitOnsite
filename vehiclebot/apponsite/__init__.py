
import logging
import logging.config
import asyncio

from vehiclebot.patches import (
    asyncio_monkey_patch,
    aiortc_monkey_patch,
    patch_asyncio_platform_loop_policy
)

#Apply patches to methods to add more functions
asyncio_monkey_patch()
aiortc_monkey_patch()
patch_asyncio_platform_loop_policy()

from decouple import config as deconf
from aiohttp.web import Application
from aiohttp.client import ClientSession

from vehiclebot.config import load_config
from vehiclebot.log import init_root_logger, aio_exc_hdl
from vehiclebot.taskmanager import TaskManager
from vehiclebot.management import init_routes

root_logger : logging.Logger = init_root_logger()

def client_session_ctx(api : dict):
    async def _wrap_client_session_ctx(app : Application):
        '''
        Create a client session object to the application
        '''
        _log = logging.getLogger('client_session_ctx')
        _log.debug('Creating ClientSession')
        setattr(app, 'cli', ClientSession(base_url=api.get('api_base')))
        yield
        _log.debug('Closing ClientSession')
        await app.cli.close()
    return _wrap_client_session_ctx
    
async def app_task_ctx(app : Application):
    _log = logging.getLogger('app_task_ctx')
    _log.debug('Creating Application tasks')
    setattr(app, 'tm', TaskManager(app))
    await app.tm.enumerate_tasks(app['cfg']['tasks'] or {})
    yield
    _log.debug('Closing Application tasks...')
    await app.tm.close()
    
async def VehicleEntryExitOnSite(loop : asyncio.AbstractEventLoop = None, debug : bool = False):
    '''The main application entry'''
    if loop is None:
        loop = asyncio.get_event_loop()

    loop.set_debug(debug)
    if debug:
        root_logger.info("Debug mode is on")

    loop.set_exception_handler(aio_exc_hdl)

    #Config data
    CFG_FILE = deconf('CONFIG', default='config.yaml')
    DISABLE_TASKS = deconf('NO_TASKS', cast=bool, default=False)

    #Load configuration data
    root_logger.info("Loading configuration file from '%s'...", CFG_FILE)
    cfg = load_config(CFG_FILE)

    #Set-up the logger from config
    root_logger.info("Setting up the logger...")
    _log_cfg = cfg.get('logger')
    if _log_cfg is not None:
        logging.config.dictConfig(_log_cfg)

    if debug:
        root_logger.setLevel(logging.DEBUG)

    root_logger.info("Starting Vehicle detection program...")
    #Application to manage all tasks and a backend access
    app = Application(loop=loop)
    app['cfg'] = cfg.get('app', {})
    root_logger.debug("AIO application created")
    
    #Management pages
    if "management" in app['cfg']:
        init_routes(app, **(app['cfg'].get('management') or {}))
        root_logger.info("Management routes added")
    
    if "integrated_model_server" in app['cfg']:
        root_logger.info("Starting integrated model server")
        from vehiclebot.appvision import VehicleVision
        integrated_server = await VehicleVision(loop=loop, debug=debug)
        app.add_subapp("/model/", integrated_server)
        app['server'] = integrated_server
        
    if "api" in app['cfg']:
        app.cleanup_ctx.append(client_session_ctx(app['cfg'].get('api')))

    #Tasks
    if not DISABLE_TASKS:
        app.cleanup_ctx.append(app_task_ctx)
    else:
        root_logger.warning("Tasks are disabled (NO_TASKS=True), hence no application tasks were loaded")

    root_logger.debug("AIO cleanup tasks created")

    root_logger.info("Application now ready to launch")
    return app
