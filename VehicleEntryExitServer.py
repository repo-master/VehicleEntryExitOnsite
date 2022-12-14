
import argparse
from vehiclebot.appvision import VehicleVision
from aiohttp.web import run_app

'''
Run app without gunicorn
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle \"Vision\" standalone server runner")
    parser.add_argument('--port', '-p', type=int, help='Port to run the HTTP server', default=5000)
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output in logs and asyncio', default=False)

    args = parser.parse_args()
    
    app = VehicleVision(debug=args.debug)
    run_app(app, port=args.port)
