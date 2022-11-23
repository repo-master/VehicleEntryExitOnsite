
from aiohttp import web
import aiohttp_jinja2
import jinja2

from .rtc import RTCServer

from typing import Dict

'''
Template functions
'''

@aiohttp_jinja2.template("dashboard.j2.html")
async def page_dashboard(request: web.Request) -> Dict[str, str]:
    return {"APIBASE": '/'}


def init_routes(app : web.Application):
    '''
    Setup Jinja2 and add the needed routes to aiohttp server
    '''

    #Static files (JS, CSS, media, etc.)
    app.add_routes([web.static('/static', 'static/')])
    
    #Setup Jinja template engine
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader('templates/')
    )
    
    app.router.add_get("/", page_dashboard)
    
    app['rtc'] = RTCServer(app)
