
import logging
import logging.config
import asyncio

from vehiclebot.patches import (
    asyncio_monkey_patch,
    aiohttp_monkey_patch
)

#Apply patches to methods to add more functions
asyncio_monkey_patch()
aiohttp_monkey_patch()

from decouple import config as deconf
from aiohttp.web import Application

from vehiclebot.config import load_config
from vehiclebot.log import init_root_logger, aio_exc_hdl

#API routes
from .routes import init_routes

root_logger : logging.Logger = init_root_logger()

CFG_FILE = deconf('CONFIG', default='config.yaml')

async def VehicleVision(loop : asyncio.AbstractEventLoop = None, debug : bool = False):
    '''Server application entry'''
    if loop is None:
        loop = asyncio.get_event_loop()

    loop.set_debug(debug)
    if debug:
        root_logger.info("Debug mode is on")

    loop.set_exception_handler(aio_exc_hdl)

    #Load configuration data
    root_logger.info("Loading configuration file...")
    cfg = load_config(CFG_FILE)

    #Set-up the logger from config
    root_logger.info("Setting up the logger...")
    logging.config.dictConfig(cfg['logger'])

    if debug:
        root_logger.setLevel(logging.DEBUG)

    root_logger.info("Starting Vehicle vision server...")
    #Application to manage all tasks and a backend access
    app = Application(loop=loop)
    app['cfg'] = cfg['app'] or {} #Beautiful symmetry
    root_logger.debug("AIO application created")
    
    #API routes
    init_routes(app)
    root_logger.info("API routes added")
    
    root_logger.info("Application now ready to launch")
    return app
