
import logging
import logging.config
import asyncio

from vehiclebot.patches import asyncio_monkey_patch

#Apply patches to methods to add more functions
asyncio_monkey_patch()

from decouple import config as deconf
from aiohttp.web import Application
from aiohttp.client import ClientSession

from vehiclebot.config import load_config
from vehiclebot.log import init_root_logger
from vehiclebot.task import TaskManager
from vehiclebot.management import init_routes

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

def app_exc_hdl(loop : asyncio.AbstractEventLoop, context):
    loop.default_exception_handler(context)

async def VehicleEntryExitOnSite(loop : asyncio.AbstractEventLoop = None, debug : bool = False):
    '''
    The main application entry
    '''
    if loop is None:
        loop = asyncio.get_event_loop()

    loop.set_debug(debug)
    root_logger.info("Debug mode is %s" % 'on' if debug else 'off')

    loop.set_exception_handler(app_exc_hdl)

    #Load configuration data
    root_logger.info("Loading configuration file...")
    cfg = load_config(CFG_FILE)

    #Set-up the logger from config
    root_logger.info("Setting up the logger...")
    logging.config.dictConfig(cfg['logger'])

    if debug:
        root_logger.setLevel(logging.DEBUG)

    root_logger.info("Starting Vehicle detection program...")
    #Application to manage all tasks and a backend access
    app = Application(loop=loop)
    setattr(app, 'cfg', cfg['app'] or {})
    root_logger.debug("AIO application created")
    
    #Tasks
    app.cleanup_ctx.append(client_session_ctx)
    app.cleanup_ctx.append(detection_task_ctx)

    root_logger.debug("AIO cleanup tasks created")

    #Management pages
    init_routes(app)
    root_logger.info("Management routes added")
    
    root_logger.info("Application now ready to launch")
    return app