
import argparse
from vehiclebot import VehicleEntryExitOnSite
from aiohttp.web import run_app
import os

'''
Run app without gunicorn
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, help='Port to run the HTTP server', default=8080)
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output in logs and asyncio', default=False)
    parser.add_argument('--nogui', '-g', action='store_true', help='Disable GUI, work in headless mode', default=False)
    
    args = parser.parse_args()

    if args.nogui:
        os.environ['HEADLESS'] = 'True'

    app = VehicleEntryExitOnSite(debug=args.debug)
    run_app(app, port=args.port)