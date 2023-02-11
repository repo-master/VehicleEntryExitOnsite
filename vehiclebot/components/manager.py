
from collections import OrderedDict
from vehiclebot.task import AIOTask
from vehiclebot.tracker import Track
from vehiclebot.model.executor import ModelExecutor
from vehiclebot.tracker.intersection import Point, doIntersect
from vehiclebot.mathutils import angle_between

import asyncio
import typing
import datetime
import uuid
import time

import logging
import threading
from fuzzywuzzy import fuzz

from contextlib import suppress
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

    def setPlateData(self, data : dict):
        if not self.plate_known or self.matchesWith(data):
            self.update(data)

    def matchesWith(self, other : typing.Union['LicensePlate', dict]) -> bool:
        other_ok = True
        if isinstance(other, LicensePlate):
            other_ok = other.plate_known
        if self.plate_known and other_ok:
            match_ratio = self.matchProbabilty(other)
            return match_ratio >= 0.9
        return False

    def matchProbabilty(self, other : typing.Union['LicensePlate', dict]) -> float:
        return fuzz.ratio(self['plate_str'], other['plate_str']) / 100.0

class Vehicle(object):
    def __init__(self, manager, task_executor : ModelExecutor, track : Track = None, possible_matches = [], gates : dict = {}):
        self.id = uuid.uuid4()
        self.first_detected_ts = datetime.datetime.now()
        self._is_active = True
        self.logger = logging.getLogger("Vehicle.%s" % self.id_short)

        self._associated_track : Track = None
        self.license_plate = LicensePlate()
        self._plate_images_queue = deque(maxlen=15)

        self._manager = manager
        self._prev_centroid = None
        self._last_good_track = None
        self._task_exec = task_executor
        self.issued_tasks = {}
        self._reset_hits_ts = None
        self._hit_check_after = time.time()
        self._ocr_do_after = time.time()
        self._possible_matched_vehicles : typing.List[typing.Tuple[Vehicle, float]] = possible_matches
        self._posHist = np.zeros((600, 600), dtype=np.float)
        self.gate_data : typing.Dict[str, dict] = {} # All gates and their attributes
        self.initGateData(gates)

        self.centroid = None
        self.movement_direction = 0.0
        self.velocity : np.ndarray = np.zeros(2)
        self.is_moving = False
        self.timings = {'last_entry': None, 'last_exit': None} #Latest entry/exit timestamps
        self.state = '' #Entry/Exit
        self.state_ts = None

        self.associated_track = track

    def initGateData(self, gates : dict):
        self.gate_data = {**gates}
        for gate in self.gate_data.values():
            gate.update({'collision': False, '_prevCol': False, 'hit': None})

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
            if hasattr(self._associated_track, '_mv_dir'):
                self.is_moving = self._associated_track._mv_dir is not None
                if self.is_moving:
                    self.movement_direction = self._associated_track._mv_dir
            if hasattr(self._associated_track, '_velocity'):
                self.velocity = self._associated_track._velocity
            if hasattr(self._associated_track, '_centroid'):
                self.centroid = np.array(self._associated_track._centroid)
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
        heatmap_task = asyncio.get_event_loop().run_in_executor(None, self._gateDetectTask)

        if self._reset_hits_ts is not None and time.time() > self._reset_hits_ts:
            self.reset_hits()

        #Check all existing vehicles if license plate is present
        #If present, remove self, reassign original vehicle with my track

        await asyncio.gather(ocr_task, heatmap_task)

    def _gateDetectTask(self):
        if self._associated_track is not None and hasattr(self._associated_track, '_centroid'):
            if self._prev_centroid is not None:
                line_traj = (self._prev_centroid, self._associated_track._centroid)
                all_hits = []
                for g_data in self.gate_data.values():
                    line_gate = g_data['coords']
                    #Green line
                    p1 = Point(*line_gate[:2])
                    q1 = Point(*line_gate[2:4])
                    #Red line
                    p2 = Point(*line_traj[0])
                    q2 = Point(*line_traj[1])
                    g_data['_prevCol'] = g_data['collision']
                    g_data['collision'] = doIntersect(p1,q1,p2,q2)
                    if g_data['collision'] and g_data['collision'] != g_data['_prevCol']:
                        g_data['hit'] = datetime.datetime.now()
                        all_hits.append(g_data)

                if len(all_hits) > 0:
                    self.entry_exit_detect()
                    self._reset_hits_ts = time.time() + 10.0

            self._prev_centroid = self._associated_track._centroid
            
        #     cv2.add(self._posHist, cpy, self._posHist)
        #     np.clip(self._posHist, 0.0, 1.0, self._posHist)
        # self._posHist *= 0.975
        # self._posHist[self._posHist<0.01] = 0

        #Check gate collision
        #for g_name in self.gate_images:
        #    gate_img, _ = self.gate_images[g_name]
        #    _, gate_img = cv2.threshold(gate_img, 0.0, 255, cv2.THRESH_BINARY)
            # _, hist = cv2.threshold(self._posHist, 0.33, 255, cv2.THRESH_BINARY)
            # th = cv2.bitwise_and(gate_img, hist)
            # cv2.imshow("th%s%s"%(self.id_short,g_name), th)
            # cv2.waitKey(1)

    def entry_exit_detect(self):
        now = datetime.datetime.now()

        if time.time() > self._hit_check_after:
            g_active = [x['hit'] is not None for x in self.gate_data.values()]
            if len(g_active) >= len(self.gate_data)*0.8: #80% of gates are hit
                ok_gates = [v for v in self.gate_data.values() if v['hit'] is not None]
                first = min(ok_gates, key=lambda x: x['id'])
                #last = max(ok_gates, key=lambda x: x['id'])
                
                #if first['id'] != last['id']:
                # Get movement direction in radians
                v_angle = self.movement_direction
                g_angle = first['normal']
                a_delta : float = angle_between(v_angle, g_angle)
                self.logger.info("Vehicle direction: %.3f rad, angle diff between gates: %.3f rad", v_angle, a_delta)

                #a_ts = first['hit']
                #b_ts = last['hit']
                #print("Hit possible entry/exit [%s]:" % self.id_short, ok_gates, "F:", first, "L:", last)
                #delta : datetime.timedelta = (b_ts - a_ts)
                new_state = None

                #if delta > datetime.timedelta(0):
                if a_delta < np.pi/2:
                    new_state = 'Entry'
                else:
                    new_state = 'Exit'

                if new_state is not None and self.state != new_state:
                    self.state = new_state
                    self.state_ts = now
                    self._hit_check_after = time.time() + 2.0
                    self.logger.info("State changed to %s", self.state)
                    self._manager.updateVehicleState(self, self.state, now)
                    if self.state == 'Entry':
                        self.timings['last_entry'] = now
                        self.timings['last_exit'] = None
                    elif self.state == 'Exit':
                        self.timings['last_exit'] = now

                    self.reset_hits()

    def reset_hits(self):
        self._reset_hits_ts = None
        for x in self.gate_data.values():
            x['hit'] = None
            #x['collision'] = False
            #x['_prevCol'] = False

    async def _ocrImageTask(self):
        can_perform_ocr = self._associated_track is not None and hasattr(self.associated_track, 'latest_frame')
        should_perform_ocr = 'ocr' not in self.issued_tasks and time.time() > self._ocr_do_after

        is_clear = False

        #blur value is actually opposite. Lower the 'blur', more blurry it is
        BLUR_THRES = 300
        BLUR_STDEV_THRES = 5e2
        MIN_WAIT_FRAMES = 2
        CHECK_FRAMES = 8
        if len(self._plate_images_queue) >= MIN_WAIT_FRAMES:
            top_latest_frames = [x[1] for x in list(self._plate_images_queue)[-CHECK_FRAMES:]]
            blurs_avg = np.average(top_latest_frames)
            blurs_stdev = np.std(top_latest_frames)
            #print(self.id_short, blurs_avg, blurs_stdev)
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
                self.license_plate.setPlateData(plate_data)
                if len(self._possible_matched_vehicles) > 0:
                    self.logger.info("There is a chance that [%s] may be same vehicle as me. Checking plate number...", ','.join([v[0].id_short for v in self._possible_matched_vehicles]))
                self._check_possible_matched_vehicles_against_me()
                self._ocr_do_after = time.time() + 10
            elif decode_error_code == 1 or decode_error_code < 0:
                delay_ocr = 1
                self._ocr_do_after = time.time() + delay_ocr
                self.logger.info("OCR seems to have failed (code %d). Performing OCR again after %.1f seconds...", decode_error_code, delay_ocr)
        except asyncio.CancelledError:
            pass
        finally:
            self.issued_tasks.pop('ocr')
            self._manager.updateVehicleListOnDashboard()

    def _check_possible_matched_vehicles_against_me(self):
        for v, p in self._possible_matched_vehicles:
            is_active = v.is_active
            self.logger.info("Checking against %s %s", v.id_short, '(it is active)' if is_active else '')
            b_pair = (self.license_plate, v.license_plate)
            if v.license_plate.plate_known:
                pass
                #self.logger.info("I have %s, %s has %s", self.license_plate['plate_str'], v.id_short, v.license_plate['plate_str'])
                ####Check if old matches new
                #if match(): merge()
            else:
                self.logger.info("%s has plate unknown", v.id_short)

        #Check for other inactive vehicles that have same plate number

        vehicles_all : typing.Set[Vehicle] = self._manager._all_vehicles.copy()
        for v_old in vehicles_all:
            if v_old.id != self.id and not v_old.is_active:
                if v_old.license_plate.plate_known:
                    is_match = self.license_plate.matchesWith(v_old.license_plate)
                    if is_match:
                        match_prob = self.license_plate.matchProbabilty(v_old.license_plate)
                        self.logger.info("Merging with vehicle id %s (%.2f%% probability)", v_old.id_short, match_prob)
                        self.state = v_old.state
                        self.state_ts = v_old.state_ts
                        self.timings.update(v_old.timings)
                        self._manager._all_vehicles.remove(v_old)
                        break

    def __eq__(self, other : object):
        if isinstance(other, Vehicle):
            if self.license_plate.plate_known and other.license_plate.plate_known:
                return self.license_plate['plate_str'] == other.license_plate['plate_str']
            elif self.associated_track is not None and other.associated_track is not None:
                return self.associated_track.id == other.associated_track.id
        return super().__eq__(other)

    def __hash__(self):
        return int(self.id)

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
                        #cv2.imshow(veh.id_short, fr)

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            cv2.waitKey(max(1,int(delaySleep*1000)))
        cv2.destroyAllWindows()

    # === End thread realm ===

    async def safeShutdown(self, timeout : float = 10):
        self._stopEv.set()
        if self.is_alive():
            await asyncio.get_event_loop().run_in_executor(None, self.join, timeout)

ENTRYEXIT_GATES = {
    #    id, x1, y1,   x2,  y2,   normal angle
    "IMG_4081.MOV": [
        (0, 1.0, 0.50, 0.0, 0.50, None)
    ],

    "00414.mp4": [
        (0, 0.3, 0.45, 1.0, 0.55, None)
    ],
    "00415.mp4": [
        (0, 0.0, 0.45, 1.0, 0.45, None)
    ],
    "00417.mp4": [
        (0, 0.0, 0.65, 1.0, 0.65, None)
    ],
    "00419.mp4": [
        (0, 0.0, 0.60, 1.0, 0.55, None)
    ],
    "00420.mp4": [
        (0, 0.0, 0.65, 1.0, 0.65, None)
    ],

    "IMG_4310.MOV": [
        (0, 0.0, 0.30, 1.0, 0.55, None)
    ]
}

class VehicleManager(AIOTask):
    def __init__(self, tm, task_name,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)

        self._stopEv = asyncio.Event()
        self._all_tracks : typing.OrderedDict[int, Track] = OrderedDict()
        self._all_vehicles : typing.Set[Vehicle] = set()
        self._task_exec : ModelExecutor = self.tm.app['model_executor']
        self.entry_exit_gates = {}

        self.vdebug = VehicleDebug(self._all_tracks, self._all_vehicles)

    async def start_task(self):
        #await asyncio.get_event_loop().run_in_executor(None, self.vdebug.start)
        pass

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
        #             candidates.append((veh, iou_region))
        #         print(veh.id_short, "bbox:", veh._last_good_track.bbox, "overlap:", iou_region, "dt:", lost_found_dt)

        candidates.extend([
            (veh, get_iou(track.bbox, veh._last_good_track.bbox))
            for veh in self._all_vehicles if (
                get_iou(track.bbox, veh._last_good_track.bbox) >= IOU_THRESH
                and (track._create_ts - veh._last_good_track._update_ts) <= LOST_FOUND_DT_THRESH
            )
        ])

        return candidates

    def updateVehicleState(self, vehicle : Vehicle, state : str, timestamp : datetime.datetime):
        asyncio.run_coroutine_threadsafe(self.sendVehicleUpdate(vehicle, state, timestamp), self.tm.loop)

    def updateVehicleListOnDashboard(self):
        asyncio.run_coroutine_threadsafe(self.sendVehicleList(), self.tm.loop)

    async def sendVehicleUpdate(self, vehicle : Vehicle, state : str, timestamp : datetime.datetime):
        await self.tm.emit('vehicledata', 'state', vehicle, state, timestamp)

    async def sendVehicleList(self):
        await self.tm.emit('vehicledata', 'broadcastChange', vehicles=self._all_vehicles)

    def initGates(self, gates : list):
        self.entry_exit_gates.clear()
        for gate_obj in gates:
            (idx, x1, y1, x2, y2, normal) = gate_obj

            if normal is None:
                #Calculate normal angle from the gate points
                l_dx = x2 - x1
                l_dy = y1 - y2
                normal = np.arctan2(l_dy, l_dx) - np.pi/2

            self.entry_exit_gates[idx] = {'id': idx, 'coords': (x1,y1,x2,y2), 'normal': normal}

    async def __call__(self):
        self.on("track", self.handleTracksUpdate)

        #DEBUG
        with suppress(KeyError):
            capTask = self.tm['camera_source']
            await self._reset_gates(capTask.source)
            capTask.on('source_changed', self._reset_gates)

    async def _reset_gates(self, src):
        #Determine camera source to select gate set
        with suppress(KeyError):
            import os
            src_file = os.path.basename(src)
            gates = ENTRYEXIT_GATES.get(src_file)
            if gates is not None:
                self.initGates(gates)

    async def handleTracksUpdate(self, tracks : typing.OrderedDict[int, Track]):
        new_tracks = tracks.keys() - self._all_tracks.keys()
        deleted_tracks = self._all_tracks.keys() - tracks.keys()
        updated_tracks = self._all_tracks.keys() & tracks.keys()

        #Insert new tracks
        for trk in new_tracks:
            self.logger.info("Found new track %d [class %d]", trk, tracks[trk].class_id)
            self._all_tracks[trk] = tracks[trk]
            #Check ID of License plate
            #if tracks[trk].class_id == 0:
            matches = self.findVehiclesThatMightBeThisTrack(tracks[trk])
            if any([x[0].is_active for x in matches]):
                max_iou = max(matches, key=lambda x: x[1])
                if max_iou[1] > 0.3:
                    self.logger.info("Track [%d, class %d] was not assigned a vehicle as it is too close to another vehicle '%s'", tracks[trk].class_id, trk, max_iou[0].id_short)
                    continue

            #Add new vehicle instance
            self._all_vehicles.add(Vehicle(
                self,
                task_executor=self._task_exec,
                track=tracks[trk],
                possible_matches=matches,
                gates=self.entry_exit_gates
            ))
            await self.tm.emit('vehicledata', 'broadcastChange', vehicles=self._all_vehicles)
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

