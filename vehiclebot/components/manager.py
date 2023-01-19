
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
from difflib import SequenceMatcher

from collections import deque
import itertools

import cv2
import numpy as np
import pickle

from motrackers.utils.misc import iou_xywh as get_iou

class LicensePlate(dict):
    def __init__(self, plate_str : str = None):
        super().__init__(plate_str=plate_str, plate_number={}, plate_raw='', detect_ts=datetime.datetime.now())
    @property
    def plate_known(self):
        return self.get('plate_str') is not None

class Vehicle(object):
    def __init__(self, task_executor : ModelExecutor, track : Track = None, possible_matches = []):
        self.id = uuid.uuid4()
        self._is_active = True
        self.logger = logging.getLogger("Vehicle.%s" % self.id_short)

        self._associated_track : Track = None
        self.license_plate = LicensePlate()
        self._plate_images_queue = deque(maxlen=15)

        self._last_good_track = None
        self._task_exec = task_executor
        self.issued_tasks = {}
        self._ocr_do_after = time.time()
        self._possible_matched_vehicles : typing.List[typing.Tuple[Vehicle, float]] = possible_matches

        self.associated_track = track

    @property
    def id_short(self) -> str:
        return self.id.hex[-6:]

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
        if self._associated_track is not None:
            self._last_good_track = self._associated_track
            if hasattr(self._associated_track, 'latest_frame'):
                self._plate_images_queue.append((
                    self._associated_track.latest_frame.copy(),
                    self._associated_track.metric_blur
                ))
                '''
                pickle.dump((
                    self._associated_track.latest_frame.copy(),
                    self._associated_track.metric_blur,
                    self._associated_track.age,
                    self._associated_track.latest_frame.shape,
                    self.id_short,
                    self._associated_track.detection_confidence
                ), open("nsb_data/gud_%s_%d.pkl" % (self.id_short, self._associated_track.age), "wb"))
                '''

    @property
    def is_active(self):
        return self._is_active and self._associated_track is not None

    async def update(self):
        ocr_task = self._ocrImageTask()

        #Check all existing vehicles if license plate is present
        #If present, remove self, reassign original vehicle with my track

        await asyncio.gather(ocr_task)

    async def _ocrImageTask(self):
        can_perform_ocr = not self.license_plate.plate_known and self._associated_track is not None and hasattr(self.associated_track, 'latest_frame')
        should_perform_ocr = 'ocr' not in self.issued_tasks and time.time() > self._ocr_do_after

        is_clear = False

        #blur value is actually opposite. Lower the 'blur', more blurry it is
        BLUR_THRES = 400
        BLUR_STDEV_THRES = 5e2
        MIN_WAIT_FRAMES = 2
        CHECK_FRAMES = 8
        if len(self._plate_images_queue) >= MIN_WAIT_FRAMES:
            top_latest_frames = [x[1] for x in list(self._plate_images_queue)[-CHECK_FRAMES:]]
            blurs_avg = np.average(top_latest_frames)
            blurs_stdev = np.std(top_latest_frames)
            is_clear = blurs_stdev < BLUR_STDEV_THRES and blurs_avg >= BLUR_THRES

        should_perform_ocr = should_perform_ocr and is_clear

        if can_perform_ocr and should_perform_ocr:
            _exec_task = self._task_exec.run("ocr", img=self.associated_track.latest_frame.copy())
            if _exec_task is not None:
                self.logger.info("Sending OCR request...")
                self.issued_tasks['ocr'] = _exec_task
                _exec_task.add_done_callback(self._ocr_done)

    def _ocr_done(self, task : asyncio.Task):
        try:
            plate_data : dict = task.result()
            if plate_data is None or not isinstance(plate_data, dict):
                self.logger.info("Task OCR failed")
                return
            #print("Plate no. is", self.license_plate['plate_str'], self.id_short, flush=True)
            self.logger.info("Task OCR finished: %s", str(plate_data))
            decode_error_code = plate_data.pop('code', None)
            if decode_error_code == 0:
                self.license_plate.update(plate_data)
                if len(self._possible_matched_vehicles) > 0:
                    self.logger.info("There is a chance that [%s] may be same vehicle as me. Checking plate number...", ','.join([v[0].id_short for v in self._possible_matched_vehicles]))
                    self._check_possible_matched_vehicles_against_me()
            elif decode_error_code == 1 or decode_error_code < 0:
                delay_ocr = 1
                self._ocr_do_after = time.time() + delay_ocr
                self.logger.info("OCR seems to have failed (code %d). Performing OCR again after %.1f seconds...", decode_error_code, delay_ocr)
        except asyncio.CancelledError:
            pass
        finally:
            self.issued_tasks.pop('ocr')

    def _check_possible_matched_vehicles_against_me(self):
        for v, p in self._possible_matched_vehicles:
            is_active = v.is_active
            self.logger.info("Checking against %s %s", v.id_short, p, '(it is active)' if is_active else '')
            b_pair = (self.license_plate, v.license_plate)
            print(b_pair)
            if v.license_plate.plate_known:
                pass
                #self.logger.info("I have %s, %s has %s", self.license_plate['plate_str'], v.id_short, v.license_plate['plate_str'])
                ####Check if old matches new
                #if match(): merge()
            else:
                self.logger.info("%s has plate unknown", v.id_short)

    def __eq__(self, other : object):
        if isinstance(other, Vehicle):
            if self.license_plate.plate_known and other.license_plate.plate_known:
                return self.license_plate['plate_str'] == other.license_plate['plate_str']
            elif self.associated_track is not None and other.associated_track is not None:
                return self.associated_track.id == other.associated_track.id
        return super().__eq__(other)

    def __hash__(self):
        if self.license_plate.plate_known:
            return self.license_plate['plate_str']
        elif self.associated_track is not None:
            return self.associated_track.id
        return self.id

    async def kill_all_tasks(self):
        await asyncio.gather(*self.issued_tasks.values())

class VehicleDebug(threading.Thread):
    def __init__(self,
                tracks_list : typing.OrderedDict[int, Track],
                vehicle_list : typing.Set[Vehicle],
                **kwargs):
        super().__init__(daemon=True)

        self._update_rate = 30.0
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
                trk = veh.associated_track or veh._last_good_track
                #print(veh.id_short, veh.license_plate['plate_str'], trk)
                if trk is not None:
                    imgs = veh._plate_images_queue.copy()
                    #pickle.dump(imgs, open("ashpak_data/vid2_%s_%d.pkl" % (veh.id_short, trk.age), "wb"))
                    if len(imgs) > 0:
                        imgs_modified = []
                        for img, blur in imgs:
                            img = cv2.resize(img, (192, 64))
                            m_txt = "{:2f}".format(blur)
                            cv2.putText(img, m_txt, (8,12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)
                            cv2.putText(img, m_txt, (8,12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
                            imgs_modified.append(img)
                        frame = np.hstack(imgs_modified) #getattr(trk, 'latest_frame', None)
                        scale = 0.8
                        h, w = frame.shape[:2]
                        fr = cv2.resize(frame, (int(w*scale), int(h*scale)))

                        plate_no = "id:{}[{}], {}, {}".format(
                            trk.id,
                            trk.class_id,
                            str(veh.license_plate['plate_str']) if veh.license_plate.plate_known else '---',
                            'active' if veh.is_active else 'inactive'
                        )
                        cv2.putText(fr, plate_no, (8,24), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,0), 3)
                        cv2.putText(fr, plate_no, (8,24), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,255,255), 2)

                        cv2.putText(fr, veh.id_short, (8,40), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,0), 3)
                        cv2.putText(fr, veh.id_short, (8,40), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,255,255), 2)
                        cv2.imshow(veh.id_short, fr)

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            cv2.waitKey(max(1,int(delaySleep*1000)))
        cv2.destroyAllWindows()

    # === End thread realm ===

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

        self.vdebug = VehicleDebug(self._all_tracks, self._all_vehicles)

    async def start_task(self):
        await asyncio.get_event_loop().run_in_executor(None, self.vdebug.start)

    async def stop_task(self):
        self._stopEv.set()
        self.logger.debug("Waiting for vehicle debug process to close...")
        await self.vdebug.safeShutdown()
        self.logger.debug("Vehicle manager process closed")
        await asyncio.gather(*[v.kill_all_tasks() for v in self._all_vehicles])
        await self.wait_task_timeout(timeout=3.0)

    def findVehiclesThatMightBeThisTrack(self, track : Track) -> typing.List[typing.Tuple[Vehicle, float]]:
        candidates : typing.List[typing.Tuple[Vehicle, float]] = []
        IOU_THRESH = 0.25
        LOST_FOUND_DT_THRESH = datetime.timedelta(seconds=3)
        #lost_vehicles = list(filter(lambda x: x.associated_track is None, self._all_vehicles))
        # print("Checking if any lost vehicles might be this track...")
        # print("Lost vehicles:", lost_vehicles)
        # print("New vehicle bbox:", track.bbox, "found time:", track._create_ts)
        # for veh in lost_vehicles:
        #     if veh._last_good_track is not None:
        #         iou_region = get_iou(track.bbox, veh._last_good_track.bbox)
        #         lost_found_dt = track._create_ts - veh._last_good_track._update_ts
        #         if iou_region >= IOU_THRESH and lost_found_dt <= LOST_FOUND_DT_THRESH:
        #             print("Might be", veh.id_short)
        #             candidates.append(veh)
        #         print(veh.id_short, "bbox:", veh._last_good_track.bbox, "overlap:", iou_region, "dt:", lost_found_dt)

        candidates.extend([
            (veh, get_iou(track.bbox, veh._last_good_track.bbox))
            for veh in self._all_vehicles if (
                get_iou(track.bbox, veh._last_good_track.bbox) >= IOU_THRESH
                and (track._create_ts - veh._last_good_track._update_ts) <= LOST_FOUND_DT_THRESH
            )
        ])

        return candidates

    async def __call__(self):
        self.on("track", self.handleTracksUpdate)

    async def handleTracksUpdate(self, tracks : typing.OrderedDict[int, Track]):
        new_tracks = tracks.keys() - self._all_tracks.keys()
        deleted_tracks = self._all_tracks.keys() - tracks.keys()
        updated_tracks = self._all_tracks.keys() & tracks.keys()

        #Insert new tracks
        for trk in new_tracks:
            self.logger.info("Found new track %d [class %d]", tracks[trk].class_id, trk)
            self._all_tracks[trk] = tracks[trk]
            #Check ID of License plate
            if tracks[trk].class_id == 1:
                matches = self.findVehiclesThatMightBeThisTrack(tracks[trk])
                if any([x[0].is_active for x in matches]):
                    max_iou = max(matches, key=lambda x: x[1])
                    if max_iou[1] > 0.3:
                        self.logger.info("Track [%d, class %d] was not assigned a vehicle as it is too close to another vehicle '%s'", tracks[trk].class_id, trk, max_iou[0].id_short)
                        continue
                self._all_vehicles.add(Vehicle(task_executor=self._task_exec, track=tracks[trk], possible_matches=matches))
        #Update existing tracks
        for trk in updated_tracks:
            self._all_tracks[trk] = tracks[trk]
            #All vehicles associated with this track
            assoc_vehicles = filter(lambda x: x.associated_track is not None and x.associated_track.id == trk, self._all_vehicles)
            for veh in assoc_vehicles:
                #Assign newly updated track (same track id, just new data).
                #We need to do this as Track object may get re-created every update due to how
                #the Process sends pickled data
                veh.associated_track = tracks[trk]
        #Delete removed tracks
        for trk in deleted_tracks:
            self.logger.info("Track %d removed", trk)
            assoc_vehicles = filter(lambda x: x.associated_track is not None and x.associated_track.id == trk, self._all_vehicles)
            for veh in assoc_vehicles:
                veh.associated_track = None
            self._all_tracks.pop(trk)

        #Update existing vehicle objects
        await asyncio.gather(*[veh.update() for veh in self._all_vehicles if veh.is_active])
        await self.tm.emit(
            'vidmixer',
            "vehicle",
            vehicles=self._all_vehicles
        )

