
import logging
import logging.config
import asyncio

from decouple import config as deconf
from aiohttp.web import Application
from aiohttp.client import ClientSession

from vehiclebot.config import load_config
from vehiclebot.log import init_root_logger
from vehiclebot.task import TaskManager

root_logger : logging.Logger = init_root_logger()

CFG_FILE = deconf('CONFIG', default='config.yaml')

async def client_session_ctx(app : Application):
    '''
    Create a client session object to the application
    '''
    _log = logging.getLogger('client_session_ctx')
    _log.debug('Creating ClientSession')
    setattr(app, 'cli', ClientSession(app.cfg['api']['api_base']))
    yield
    _log.debug('Closing ClientSession')
    await app.cli.close()
    
async def detection_task_ctx(app : Application):
    _log = logging.getLogger('detection_task_ctx')
    _log.debug('Creating Vehicle detection tasks')
    setattr(app, 'tm', TaskManager(app))
    await app.tm.enumerate_tasks(app.cfg['tasks'] or {})
    yield
    _log.debug('Closing detection tasks')
    await app.tm.close()

async def VehicleEntryExitOnSite(loop : asyncio.AbstractEventLoop = None):
    '''
    The main application entry
    '''
    if loop is None:
        loop = asyncio.get_event_loop()

    #Load configuration data
    root_logger.info("Loading configuration file...")
    cfg = load_config(CFG_FILE)

    #Set-up the logger from config
    root_logger.info("Setting up the logger...")
    logging.config.dictConfig(cfg['logger'])

    root_logger.info("Starting Vehicle detection program...")
    #Application to manage all tasks and a backend access
    app = Application(loop=loop)
    setattr(app, 'cfg', cfg['app'] or {})
    root_logger.debug("AIO application created")
    
    #Tasks
    app.cleanup_ctx.append(client_session_ctx)
    app.cleanup_ctx.append(detection_task_ctx)

    root_logger.debug("AIO cleanup tasks created")
    
    root_logger.info("Application now ready to launch")
    return app