
import argparse
from vehiclebot import VehicleEntryExitOnSite
from aiohttp.web import run_app

'''
Run app without gunicorn
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, help='Port to run the HTTP server', default=8080)

    args = parser.parse_args()
    app = VehicleEntryExitOnSite()
    run_app(app, port=args.port)