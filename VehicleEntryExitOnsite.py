
import argparse
from vehiclebot.apponsite import VehicleEntryExitOnSite
from aiohttp.web import run_app
import os

######
##TODO
######
#
# [ ] More models in server: [x] Plate detect, [x] OCR (combine both), [ ] facing side (front/back/none)
# [ ] Make the remote detector use another class as delegate, so that RemoteDetector is
#     actually able to use any class to get detection (local or not). It becomes "ObjectDetector"
#     and "PlateRecogniser" and they use other class objects such as "RemoteDetector" and "LocalDetector"
#[ ] Sort vehicle detections in dashboard most recent on top


'''
PLANNED
------------
- SQLite Database locally to store:
  - All tracks, with sequence number to keep track id
  - Use peewee
- Temporary track ID before using the database sequence after successful track (age more than 10)
- Associate all tracks with known license plate from a table, with likelyhood estimate of each
- Track IDs must be unique even between runs of the app, so use UUID (use monkey patch on the tracker)

'''


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
