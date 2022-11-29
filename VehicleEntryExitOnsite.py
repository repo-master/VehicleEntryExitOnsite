
import argparse
from vehiclebot.apponsite import VehicleEntryExitOnSite
from aiohttp.web import run_app
import os

######
##TODO
######
#
# [x] Bring back trajectory, but now using instance of Track than separate dict
# [ ] Also bring back DataChannel to show on dashboard
# [x] If 'lost' > 0, then update featuretracker bbox into motracker bbox till not lost
# [ ] Feature tracker should be moved smoothly
# [x] Show ghost rectangle of featuretracker bbox
# [ ] More models in server: Plate detect, OCR (combine both), facing side (front/back/none)
# [ ] Make the remote detector use another class as delegate, so that RemoteDetector is
#     actually able to use any class to get detection (local or not). It becomes "ObjectDetector"
#     and "PlateRecogniser" and they use other class objects such as "RemoteDetector" and "LocalDetector"

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
