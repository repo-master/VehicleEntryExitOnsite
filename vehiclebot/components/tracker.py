
from ..task import AIOTask, TaskOrTasks
from ..multitask import AsyncProcess

from .camera import CameraSource
from ..types import Detections

from ..tracker import Trajectory, Tracker, Track, motrackers, TRACK_COLORS

from ..imutils import cropImage, scaleImgRes

import cv2
import time
import queue
import typing
import asyncio
import threading
import collections
from datetime import datetime, timedelta
from contextlib import suppress

import numpy as np

import pickle

#TODO: Use EKF to combine both boxes from featuretracker and detector

class FeatureTracker:
    '''
    Uses OpenCV's single-object tracker to track quickly the bounding boxes
    detected by the multi-object tracker
    '''
    def __init__(self, tracker : Tracker, featuretracker_factory = cv2.TrackerCSRT):
        self._lastFrame : np.ndarray = None
        self._tracker = tracker
        self._trackfact = featuretracker_factory
        self._newFrameReceived = False

    def trackerID(self):
        return "ft_%d" % id(self)
    
    def updateTrackFrame(self, update_frame : np.ndarray):
        '''Gives tracker new image to update current tracks'''
        self._lastFrame = update_frame
        #TODO: Better way to do this, maybe queue
        self._newFrameReceived = True

    @staticmethod
    def validateBBox(bbox : np.ndarray, image_shape : np.ndarray) -> bool:
        return (
            np.prod(bbox.shape[2:])>0 #BBox should not be size 0 in any dimension
            and all((bbox[2:]+bbox[:2])<=image_shape[1::-1]) #BBox should not fall outside of the image
        )

    def updateDetection(self, det_frame : np.ndarray):
        '''Update tracking features from the frame to use as template'''
        if det_frame is None:
            return
        
        #Scale image according to scale factor
        scale = getattr(self._tracker, "_scale", 1.0)
        frame_scaled, _ = scaleImgRes(det_frame, scale=scale)

        for trk_obj in self._tracker.tracks.values():
            _tracker_id = self.trackerID()
            if not hasattr(trk_obj, _tracker_id):
                setattr(trk_obj, _tracker_id, self._trackfact.create())
                
            setattr(trk_obj, "%s_bbox" % _tracker_id, tuple(trk_obj.bbox))
            setattr(trk_obj, '_is_lost', trk_obj.lost > 20)

            can_update_tracker = (
                trk_obj.lost == 0 #If data of tracker is from a good detection
                and self.validateBBox(trk_obj.bbox, frame_scaled.shape)
                #TODO: Check time period
            )

            if can_update_tracker:
                #Init feature tracker with new frame
                ftrack = getattr(trk_obj, _tracker_id)
                ftrack.init(frame_scaled, np.clip(trk_obj.bbox, (0, 0, 0, 0), np.resize(frame_scaled.shape[1::-1], 4)).astype(int))
                
    def getBBox(self, trk_obj) -> np.ndarray:
        _tracker_id = self.trackerID()
        if hasattr(trk_obj, "%s_bbox" % _tracker_id):
            return getattr(trk_obj, "%s_bbox" % _tracker_id)
    
    def update(self):
        '''Update the trackers' bbox by feature matching current detection with current updated frame'''
        if self._lastFrame is None or not self._newFrameReceived:
            return

        self._newFrameReceived = False
        _tracker_id = self.trackerID()

        scale = getattr(self._tracker, "_scale", 1.0)
        frame_scaled, _ = scaleImgRes(self._lastFrame, scale=scale)

        for trk_obj in self._tracker.tracks.values():
            if hasattr(trk_obj, _tracker_id):
                ftrack = getattr(trk_obj, _tracker_id)
                if self.validateBBox(trk_obj.bbox, frame_scaled.shape):
                    success, new_bbox = ftrack.update(frame_scaled)
                    if success:
                        setattr(trk_obj, "%s_bbox" % _tracker_id, new_bbox)

                    needs_tracker_nudge = (
                        trk_obj._is_lost or
                        (datetime.now()-trk_obj._update_ts) > timedelta(seconds=2.0)
                    )
                    if needs_tracker_nudge:
                        #Nudge the tracker bbox towards the feature bbox just so that detection could be possible
                        #TODO: only move it just little using tween
                        trk_obj.bbox = np.array(new_bbox)

    __call__ = update
    
class TrackerProcess(threading.Thread):
    def __init__(self, tracker_config : dict):
        super().__init__(daemon=True)
        self._update_rate : float = tracker_config.pop('update_rate', 10.0)

        #Load tracker
        _tracker_type = tracker_config.pop('type')
        tracker_cls : typing.Type[Tracker] = getattr(motrackers, _tracker_type) #Throws AttributeError if invalid
        if not issubclass(tracker_cls, Tracker):
            raise ValueError("Tracker type '%s' is not a valid tracker!" % str(_tracker_type))
        
        self.tracker : Tracker = tracker_cls(
            tracker_output_format='mot_challenge',
            **tracker_config
        )

        self._track_only_classes : typing.List = tracker_config.pop('track_classes')
        if self._track_only_classes is not None and not isinstance(self._track_only_classes, typing.Iterable):
            self._track_only_classes = [self._track_only_classes]

        self._feature_tracker = FeatureTracker(self.tracker)
        #Trajectory estimator (direction, speed, etc.)
        self._trajectory = Trajectory(self.tracker)

        self._stopEv = threading.Event()
        self._detections : queue.Queue[typing.Tuple[Detections, np.ndarray, float]] = queue.Queue(maxsize=10)
        self._trackresults : typing.Deque[typing.Dict] = collections.deque(maxlen=200)

        #Start thread
        self.start()

    def stop(self, timeout : float = None):
        self._stopEv.set()
        self.join(timeout=timeout)

    def cleanup(self):
        pass

    def run(self):
        if self._update_rate is None:
            self._update_rate = 60

        next_time = time.time()
        delaySleep = 0
        while not self._stopEv.wait(timeout=delaySleep):
            try:
                #Any new detections?
                #Update trackers and predict new (estimated) position
                self._updateTracker(*self._detections.get_nowait())
            except queue.Empty:
                pass
                
            #Update tracker that moves the tracks using the detected feature
            #self._feature_tracker()
            self._trajectory(None)#self._feature_tracker.getBBox)
            
            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            if delaySleep < 0:
                delaySleep = 0
                next_time = time.time()

        self.cleanup()

    def _updateTracker(self, dets : Detections, frame : np.ndarray, scale : float):
        self.tracker.update(*dets)
        self._updateTrackerStats(scale)
        #self._feature_tracker.updateDetection(frame)
        self._updateResults(frame, scale)
        
    def _updateTrackerStats(self, scale : float):
        setattr(self.tracker, '_scale', scale)
        for trk_obj in self.tracker.tracks.values():
            setattr(trk_obj, '_update_ts', datetime.now())
            if trk_obj.age == 1:
                setattr(trk_obj, '_create_ts', trk_obj._update_ts)
                
    def _updateResults(self, frame : np.ndarray, scale : float):
        results = []
        inv_scale = 1/scale
        _trj_id = self._trajectory.trackerID()
        for trk_obj in self.tracker.tracks.values():
            trk = trk_obj.output()
            trj_obj = getattr(trk_obj, _trj_id, None)

            #Extract all information needed
            trk_id = trk[1]
            xmin, ymin, width, height = trk[2:6]
            conf : float = trk[6]
            class_id : int = trk_obj.class_id

            detect_ts, update_ts = None, None
            traj_dir_angle = None
            traj_dir_mv = None
            traj_mv_spd = None
            traj_is_mv = None
            traj_pos = None
            
            if hasattr(trk_obj, '_create_ts'):
                detect_ts = trk_obj._create_ts
            if hasattr(trk_obj, '_update_ts'):
                update_ts = trk_obj._update_ts

            if trj_obj is not None:
                traj_dir_angle = trj_obj['dir']
                traj_dir_mv = trj_obj['cardinal']
                traj_mv_spd = trj_obj['speed']
                traj_is_mv = trj_obj['is_moving']
                traj_pos = np.array(trj_obj['curr_pos'])*inv_scale

            #Calculate centroid of current detection
            xcentroid, ycentroid = xmin + 0.5*width, ymin + 0.5*height

            #Integer for cropping
            xmin, ymin, width, height = int(xmin*inv_scale), int(ymin*inv_scale), int(width*inv_scale), int(height*inv_scale)
            xcentroid, ycentroid = int(xcentroid*inv_scale), int(ycentroid*inv_scale)
            
            img_crop = cropImage(frame, xmin, ymin, xmin+width, ymin+height)

            if img_crop.shape[0]*img_crop.shape[1] == 0: continue

            #TODO: Class Object
            res_item = {
                'track_id': trk_id,
                'age': trk_obj.age,
                'lost_for_frames': trk_obj.lost,
                'img': img_crop,
                'detection_class': (class_id, conf),
                'is_new_detection': trk_obj.age == 1,
                'bbox': (xmin, ymin, width, height),
                'first_detect_ts': detect_ts,
                'last_update_ts': update_ts,
                'movement_speed': traj_mv_spd,
                'is_moving': traj_is_mv,
                'track_pos': traj_pos,
                'centroid_pos': (xcentroid, ycentroid),
                'estimated_movement_angle': traj_dir_angle,
                'estimated_movement_direction': traj_dir_mv
            }
            results.append(res_item)

        self._trackresults.append(results)

    #Public methods

    def updateVideoFrame(self, frame : np.ndarray):
        '''Update feature tracker with new video frame'''
        self._feature_tracker.updateTrackFrame(frame)
        
    def updateDetection(self, detection : Detections, frame : np.ndarray, scale : float):
        '''Add new detection to queue for processing'''
        #NEW [also FIXME]: Filter detection by ID to only track plate
        if self._track_only_classes is not None:
            selected_ids = np.isin(detection.class_ids, self._track_only_classes)
            detection = Detections(*(x[selected_ids] for x in detection))
        self._detections.put((detection, frame, scale))
        
    def drawTrackBBoxes(self, img : np.ndarray) -> np.ndarray:
        '''Draw bboxes of current tracks on image'''
        _tracker_id = self._feature_tracker.trackerID()
        output_tracks = list(self.tracker.tracks.values()) #So that we can get all tracks instead of active ones
        for trk_obj in output_tracks:
            scale = getattr(self.tracker, "_scale", 1.0)
            inv_scale = 1/scale
            trk = trk_obj.output()
            ft_box = None
            if hasattr(trk_obj, "%s_bbox" % _tracker_id):
                bbox : tuple = getattr(trk_obj, "%s_bbox" % _tracker_id)
                ft_box = (
                    (int(bbox[0]*inv_scale), int(bbox[1]*inv_scale)),
                    (int((bbox[0]+bbox[2])*inv_scale), int((bbox[1]+bbox[3])*inv_scale))
                )
            
            can_draw_lost = False

            #Extract all information to display
            trk_id = trk[1]
            is_lost = trk_obj.lost>0
            xmin, ymin, width, height = trk[2:6]
            trk_color = TRACK_COLORS[
                -1 if is_lost else
                (trk_id % (len(TRACK_COLORS)-1))
            ]
            conf = trk[6]

            #Calculate centroid of current detection
            xcentroid, ycentroid = xmin + 0.5*width, ymin + 0.5*height

            #Integer for drawing
            xmin, ymin, width, height = int(xmin*inv_scale), int(ymin*inv_scale), int(width*inv_scale), int(height*inv_scale)
            xcentroid, ycentroid = int(xcentroid*inv_scale), int(ycentroid*inv_scale)
        
            txt_str = "{confidence:.2f}%".format(confidence=conf*100)
            trk_text = "ID {}".format(trk_id) + (' LOST' if is_lost and can_draw_lost else '')
            
            ##Drawing
            #Bounding box
            #if ft_box is not None:
            #    cv2.rectangle(img, *ft_box, (100,200,200), 1)

            if can_draw_lost or (not can_draw_lost and not is_lost):
                if ft_box is not None:
                    cv2.rectangle(img, *ft_box, trk_color, 2)
                cv2.rectangle(img, (xmin, ymin), (xmin+width, ymin+height), trk_color, 2)

            #Class and confidence
            txt_pos = [xmin, ymin]
            txt_pos[0] = max(txt_pos[0], 2)
            txt_pos[1] = max(txt_pos[1]-8, 10)
            cv2.putText(img, txt_str, txt_pos, cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
            cv2.putText(img, txt_str, txt_pos, cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 255, 255), lineType=cv2.LINE_AA, thickness=1)
            #Track id
            cv2.putText(img, trk_text, (xcentroid - 10, ycentroid - height//2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.5 if is_lost else 1.2, (0, 0, 0), lineType=cv2.LINE_AA, thickness=4)
            cv2.putText(img, trk_text, (xcentroid - 10, ycentroid - height//2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.5 if is_lost else 1.2, trk_color, lineType=cv2.LINE_AA, thickness=2)
            #Track center
            cv2.circle(img, (xcentroid, ycentroid), 4, trk_color, -1)
            
            self._trajectory.draw(trk_obj, img, inv_scale)

        return img

    def getTracks(self, max_items : int = 10):
        res = []
        for _ in range(max_items):
            try:
                res.append(self._trackresults.pop())
            except IndexError:
                break
        return res


import multiprocessing as mp
import aioprocessing as amp

class AsyncObjectTrackerProcess(mp.Process):
    def __init__(self, mot_params : dict, **kwargs):
        super().__init__()
        #How often to update the tracker
        self._update_rate : float = mot_params.pop('update_rate', 10.0)
        self._track_only_classes = [1]

        #Load tracker
        _tracker_type = mot_params.pop('type')
        #Get class of the tracker we want to use from the motrackers module
        tracker_cls : typing.Type[Tracker] = getattr(motrackers, _tracker_type) #Throws AttributeError if invalid
        if not issubclass(tracker_cls, Tracker):
            raise ValueError("Tracker type '%s' is not a valid tracker!" % str(_tracker_type))

        self.tracker : Tracker = tracker_cls(
            tracker_output_format='mot_challenge',
            **mot_params
        )

        self._stopEv : amp.Event = amp.AioEvent(context=mp.get_context())
        self._detections : amp.Queue[typing.Tuple[Detections, np.ndarray, float]] = amp.AioQueue(maxsize=10, context=mp.get_context())
        self._detectionQueue : amp.Queue[typing.OrderedDict] = amp.AioQueue(maxsize = 0, context=mp.get_context())


    #==== Process realm ====

    def run(self):
        import signal
        #Ignore SIGINT (KeyboardInterrupt)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        next_time = time.time()
        delaySleep = 0
        while not self._stopEv.wait(timeout=delaySleep):
            try:
                #Any new detections?
                #Update trackers and predict new (estimated) position
                self._updateTracker(*self._detections.get_nowait())
            except queue.Empty:
                pass

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            if delaySleep < 0:
                delaySleep = 0
                next_time = time.time()

    def _updateTracker(self, dets : Detections, frame : np.ndarray, scale : float):
        self.tracker.update(*dets)
        self._updateTrackerStats(scale)
        #self._feature_tracker.updateDetection(frame)
        self._updateResults(frame, scale)

        with suppress(queue.Full):
            self._detectionQueue.put_nowait(self.tracker.tracks)

    def _updateTrackerStats(self, scale : float):
        setattr(self.tracker, '_scale', scale)
        for trk_obj in self.tracker.tracks.values():
            setattr(trk_obj, '_update_ts', datetime.now())
            if trk_obj.age == 1:
                setattr(trk_obj, '_create_ts', trk_obj._update_ts)
                
    def _updateResults(self, frame : np.ndarray, scale : float):
        inv_scale = 1/scale
        #_trj_id = self._trajectory.trackerID()
        for trk_obj in self.tracker.tracks.values():
            trk = trk_obj.output()
            #trj_obj = getattr(trk_obj, _trj_id, None)

            #Extract all information needed
            trk_id = trk[1]
            xmin, ymin, width, height = trk[2:6]
            conf : float = trk[6]
            class_id : int = trk_obj.class_id

            '''
            detect_ts, update_ts = None, None
            traj_dir_angle = None
            traj_dir_mv = None
            traj_mv_spd = None
            traj_is_mv = None
            traj_pos = None
            
            if hasattr(trk_obj, '_create_ts'):
                detect_ts = trk_obj._create_ts
            if hasattr(trk_obj, '_update_ts'):
                update_ts = trk_obj._update_ts
            if trj_obj is not None:
                traj_dir_angle = trj_obj['dir']
                traj_dir_mv = trj_obj['cardinal']
                traj_mv_spd = trj_obj['speed']
                traj_is_mv = trj_obj['is_moving']
                traj_pos = np.array(trj_obj['curr_pos'])*inv_scale
            '''
            #Calculate centroid of current detection
            xcentroid, ycentroid = xmin + 0.5*width, ymin + 0.5*height

            #Integer for cropping
            xmin, ymin, width, height = int(xmin*inv_scale), int(ymin*inv_scale), int(width*inv_scale), int(height*inv_scale)
            xcentroid, ycentroid = int(xcentroid*inv_scale), int(ycentroid*inv_scale)
            
            img_crop = cropImage(frame, xmin, ymin, xmin+width, ymin+height)
            if img_crop.shape[0]*img_crop.shape[1] > 0:
                setattr(trk_obj, 'latest_frame', img_crop)

            #TODO: Class Object
            '''res_item = {
                'track_id': trk_id,
                'age': trk_obj.age,
                'lost_for_frames': trk_obj.lost,
                'img': img_crop,
                'detection_class': (class_id, conf),
                'is_new_detection': trk_obj.age == 1,
                'bbox': (xmin, ymin, width, height),
                'first_detect_ts': detect_ts,
                'last_update_ts': update_ts,
                'movement_speed': traj_mv_spd,
                'is_moving': traj_is_mv,
                'track_pos': traj_pos,
                'centroid_pos': (xcentroid, ycentroid),
                'estimated_movement_angle': traj_dir_angle,
                'estimated_movement_direction': traj_dir_mv
            }'''


    # === End process realm ===


    #Async interface

    async def safeShutdown(self, timeout : float = 10):
        self._stopEv.set()

        # N.B. Not using AioProcess as it is very difficult to subclass it, so manually perform shutdown
        await asyncio.get_event_loop().run_in_executor(None, self.join, timeout)
        if self.is_alive():
            await asyncio.get_event_loop().run_in_executor(None, self.terminate)
        await asyncio.get_event_loop().run_in_executor(None, self.close)

    async def updateDetection(self, detection : Detections, frame : np.ndarray, scale : float):
        '''Add new detection to queue for processing'''
        #NEW [also FIXME]: Filter detection by ID to only track plate
        if self._track_only_classes is not None:
            ## N.B.: Always run CPU-bound tasks in an executor
            selected_ids = await asyncio.get_event_loop().run_in_executor(None, np.isin, detection.class_ids, self._track_only_classes)
            detection = Detections(*(x[selected_ids] for x in detection))
        await self._detections.coro_put((detection, frame, scale))

    async def tracksUpdated(self):
        while not self._stopEv.is_set():
            with suppress(queue.Empty):
                yield await self._detectionQueue.coro_get(block=True, timeout=1.0)


class ObjectTracker(AIOTask):
    def __init__(self, tm, task_name,
                 input_source : str,
                 tracker: typing.Dict,
                 output : TaskOrTasks = None,
                 track_output : TaskOrTasks = None,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.inp_src = input_source
        self.output_dest = output
        self.tracker_params = tracker
        self.track_output_to_tasks = track_output

        self._long_running_tasks = []

        if self.tracker_params is None:
            self.logger.warning("Field 'tracker' is not given in the config for %s. Using default settings for the MOT" % self.name)
            self.tracker_params = {}

        self.tracker = AsyncObjectTrackerProcess(self.tracker_params)
        self.logger.info("Started Tracker task")
        
    async def start_task(self):
        await asyncio.get_event_loop().run_in_executor(None, self.tracker.start)
        self._long_running_tasks.append(asyncio.create_task(self.tracksProcessor()))

    async def stop_task(self):
        #Shut down long-running (polling) tasks
        self.logger.info("Closing long-running subtasks...")
        tasks = asyncio.gather(*self._long_running_tasks)
        tasks.cancel()

        self.logger.debug("Waiting for tracker process to close...")
        await self.tracker.safeShutdown()
        self.logger.debug("Tracker process closed")
        #Tasks will be cancelled so ignore cancelled exception and wait for them to close
        #This is done here as some tasks may rely on the stopEv
        with suppress(asyncio.CancelledError):
            await tasks
        await self.wait_task_timeout(timeout=3.0)

    async def __call__(self):
        try:
            capTask : CameraSource = self.tm[self.inp_src]
        except KeyError:
            self.logger.error("Input source component \"%s\" is not loaded. Please check your config file to make sure it is properly configured." % self.inp_src)
            return
        
        capTask.on('frame', self.processFrame)
        self.on('detect', self.updateDetection)

    async def tracksProcessor(self):
        '''
        Task to get current tracks as they get updated, and send the current tracks to another task
        '''
        async for tracks in self.tracker.tracksUpdated():
            await self.tm.emit(
                self.track_output_to_tasks,
                "track",
                tracks
            )

    #Input methods
    async def processFrame(self, img):
        if img is not None:
            pass
            #await self.tracker.updateVideoFrame(img)

    async def updateDetection(self, detection : Detections, frame : np.ndarray, scale : float = 1.0):
        '''Update the tracker with new set of detections'''
        if self.tracker is None: return
        try:
            await asyncio.wait_for(self.tracker.updateDetection(detection, frame, scale), timeout = 2.0)
        except asyncio.TimeoutError:
            #Skip this detection, as queue is full or not ready
            pass
