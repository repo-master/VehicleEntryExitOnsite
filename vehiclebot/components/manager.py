
from collections import OrderedDict
from vehiclebot.task import AIOTask
from vehiclebot.tracker import Track
from vehiclebot.model.executor import ModelExecutor

import asyncio
import typing
import datetime
import uuid
import time

import logging
import threading

import cv2

class LicensePlate(object):
    def __init__(self, plate_number : str = None):
        self.plate_number = plate_number
    @property
    def plate_known(self):
        return self.plate_number is not None

class Vehicle(object):
    def __init__(self, task_executor : ModelExecutor, track : Track = None):
        self.id = uuid.uuid4()
        self._is_active = True
        self.logger = logging.getLogger("Vehicle.%s" % self.id.hex[-6:])

        self._associated_track : Track = None
        self.license_plate = LicensePlate()

        self.associated_track = track
        self._task_exec = task_executor
        self.issued_tasks = {}

    @property
    def associated_track(self):
        return self._associated_track
    @associated_track.setter
    def associated_track(self, track : Track):
        if track is None:
            self.logger.info("No longer being tracked")
        elif (self._associated_track is None or self._associated_track.id != track.id):
            self.logger.info("Focused on track ID %d", track.id)

        self._associated_track = track

    @property
    def is_active(self):
        return self._is_active

    async def update(self):
        ocr_task = self._ocrImageTask()

        await asyncio.gather(ocr_task)

    async def _ocrImageTask(self):
        can_perform_ocr = not self.license_plate.plate_known and self._associated_track is not None and hasattr(self.associated_track, 'latest_frame')
        should_perform_ocr = 'ocr' not in self.issued_tasks

        if can_perform_ocr and should_perform_ocr:
            ocr_task = {
                "task": "ocr",
                "params": {
                    "img": self.associated_track.latest_frame
                }
            }
            _exec_task = self._task_exec.run(ocr_task)
            if _exec_task is not None:
                self.issued_tasks['ocr'] = _exec_task
                _exec_task.add_done_callback(self._ocr_done)

    def _ocr_done(self, task : asyncio.Task):
        print("Task OCR finished:", task.result())

    def __eq__(self, other : object):
        if isinstance(other, Vehicle):
            if self.license_plate.plate_known and other.license_plate.plate_known:
                return self.license_plate.plate_number == other.license_plate.plate_number
            elif self.associated_track is not None and other.associated_track is not None:
                return self.associated_track.id == other.associated_track.id
        return super().__eq__(other)

    def __hash__(self):
        if self.license_plate.plate_known:
            return self.license_plate.plate_number
        elif self.associated_track is not None:
            return self.associated_track.id
        return self.id

class AsyncVehicleManagerThread(threading.Thread):
    def __init__(self,
                tracks_list : typing.OrderedDict[int, Track],
                vehicle_list : typing.Set[Vehicle],
                **kwargs):
        super().__init__()

        self._update_rate = 100.0
        self._synced_tracks = tracks_list
        self._synced_vehicles = vehicle_list

        self._stopEv = threading.Event()
        self.logger = logging.getLogger(__name__)


    #==== Thread realm ====

    def run(self):
        next_time = time.time()
        delaySleep = 0
        while not self._stopEv.is_set():
            vehicles = self._synced_vehicles.copy()
            for veh in vehicles:
                trk = veh.associated_track
                if trk is not None:
                    frame = getattr(trk, 'latest_frame', None)
                    if frame is not None:
                        cv2.imshow(veh.id.hex, frame)

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            cv2.waitKey(max(1,int(delaySleep*1000)))
        cv2.destroyAllWindows()

    # === End thread realm ===

    #Async interface

    async def safeShutdown(self, timeout : float = 10):
        self._stopEv.set()
        await asyncio.get_event_loop().run_in_executor(None, self.join, timeout)

class VehicleManager(AIOTask):
    def __init__(self, tm, task_name,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)

        self._stopEv = asyncio.Event()
        self._all_tracks : typing.OrderedDict[int, Track] = OrderedDict()
        self._all_vehicles : typing.Set[Vehicle] = set()
        self._task_exec : ModelExecutor = self.tm.app['model_executor']

        self.vmanager = AsyncVehicleManagerThread(self._all_tracks, self._all_vehicles)

    async def start_task(self):
        await asyncio.get_event_loop().run_in_executor(None, self.vmanager.start)

    async def stop_task(self):
        self._stopEv.set()
        self.logger.debug("Waiting for vehicle manager process to close...")
        await self.vmanager.safeShutdown()
        self.logger.debug("Vehicle manager process closed")
        await self.wait_task_timeout(timeout=3.0)

    async def handleTracksUpdate(self, tracks : typing.OrderedDict):
        await self.updateTracks(tracks)

    async def __call__(self):
        self.on("track", self.handleTracksUpdate)

    async def updateTracks(self, tracks : typing.OrderedDict[int, Track]):
        new_tracks = tracks.keys() - self._all_tracks.keys()
        deleted_tracks = self._all_tracks.keys() - tracks.keys()
        updated_tracks = self._all_tracks.keys() & tracks.keys()

        #Insert new tracks
        for trk in new_tracks:
            self.logger.info("Found new track: %d", trk)
            self._all_tracks[trk] = tracks[trk]
            self._all_vehicles.add(Vehicle(task_executor=self._task_exec, track=tracks[trk]))
        #Update existing tracks
        for trk in updated_tracks:
            self._all_tracks[trk] = tracks[trk]
            #All vehicles associated with this track
            assoc_vehicles = filter(lambda x: x.associated_track is not None and x.associated_track.id == trk, self._all_vehicles)
            for veh in assoc_vehicles:
                #Assign newly updated track (same track id, just new data)
                veh.associated_track = tracks[trk]
        #Delete removed tracks
        for trk in deleted_tracks:
            assoc_vehicles = filter(lambda x: x.associated_track is not None and x.associated_track.id == trk, self._all_vehicles)
            for veh in assoc_vehicles:
                veh.associated_track = None
            self._all_tracks.pop(trk)
            self.logger.info("Track %d removed", trk)

        #Update existing vehicle objects
        await asyncio.gather(*[veh.update() for veh in self._all_vehicles if veh.is_active])

