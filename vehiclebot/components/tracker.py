
from ..task import AIOTask, TaskOrTasks

from .camera import CameraSource
from ..types import Detections
from ..model.filter import unsharp_mask, filter__removeShadow
from ..mathutils import point_angle
from math import dist

from ..tracker import Tracker, Track, motrackers, TRACK_COLORS

from ..imutils import cropImage, scaleImgRes

import cv2
import time
import queue
import typing
import asyncio
import collections
from datetime import datetime, timedelta
from contextlib import suppress

import numpy as np

import multiprocessing as mp
import aioprocessing as amp

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


class ObjectTrackerProcess(mp.Process):
    def __init__(self,
                stopEv : mp.Event,
                readyEv : mp.Event,
                detections : mp.Queue,
                detectionQueue : mp.Queue,
                update_rate : float,
                tracker_cls : typing.Type[Tracker],
                mot_params : dict):
        super().__init__()
        self._stopEv = stopEv
        self._ready = readyEv
        self._detections = detections
        self._detectionQueue = detectionQueue
        self._update_rate = update_rate
        self.tracker : Tracker = tracker_cls(
            tracker_output_format='mot_challenge',
            **mot_params
        )

    #==== Process realm ====

    def run(self):
        import signal
        #Ignore SIGINT (KeyboardInterrupt)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        next_time = time.time()
        delaySleep = 0
        self._ready.set()
        while not self._stopEv.wait(timeout=delaySleep):
            with suppress(queue.Empty):
                #Any new detections?
                #Update trackers and predict new (estimated) position
                self._updateTracker(*self._detections.get_nowait())

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            if delaySleep < 0:
                delaySleep = 0
                #next_time = time.time()

    def _updateTracker(self, dets : Detections, frame : np.ndarray, scale : float):
        self.tracker.update(*dets)
        self._updateTrackerStats()
        self._updateResults(frame)

        with suppress(queue.Full):
            self._detectionQueue.put_nowait(self.tracker.tracks)

    def _updateTrackerStats(self):
        for trk_obj in self.tracker.tracks.values():
            setattr(trk_obj, '_update_ts', datetime.now())
            if trk_obj.age == 1:
                setattr(trk_obj, '_create_ts', trk_obj._update_ts)
            if not hasattr(trk_obj, '_traj'):
                setattr(trk_obj, '_traj', collections.deque(maxlen=10))
            if not hasattr(trk_obj, '_mv_dir'):
                setattr(trk_obj, '_mv_dir', None)
            if not hasattr(trk_obj, '_velocity'):
                setattr(trk_obj, '_velocity', np.zeros(2))

    def _updateResults(self, frame : np.ndarray):
        inv_scale = np.resize(frame.shape[1::-1], 4)
        for trk_obj in self.tracker.tracks.values():
            bbox = trk_obj.bbox * inv_scale

            #Calculate centroid of current detection
            xcentroid, ycentroid = (trk_obj.bbox[:2] + trk_obj.bbox[2:]*0.5)

            setattr(trk_obj, '_centroid', (xcentroid, ycentroid))
            trk_obj._traj.append((xcentroid, ycentroid))
            if len(trk_obj._traj) >= 3:
                mv_pt0, mv_pt1 = trk_obj._traj[-1], trk_obj._traj[-3]
                mv_pt0 = (mv_pt0[0], 1-mv_pt0[1])
                mv_pt1 = (mv_pt1[0], 1-mv_pt1[1])
                ldist = dist(mv_pt0, mv_pt1)
                if ldist >= 1e-4:
                    t_angle = point_angle(mv_pt0, mv_pt1)
                    setattr(trk_obj, '_mv_dir', t_angle)
                    vel_vec_new = np.array([np.cos(t_angle), -np.sin(t_angle)]) * ldist * 10
                    setattr(trk_obj, '_velocity', vel_vec_new)
                else:
                    setattr(trk_obj, '_mv_dir', None)
                    setattr(trk_obj, '_velocity', np.zeros(2))

            '''padding = 0
            min_dim = min(width, height)
            if min_dim > 16:
                padding = min_dim / 8
                xmin -= padding
                ymin -= padding
                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                width += padding*2
                height += padding*2'''

            #Integer for cropping
            xmin, ymin, width, height = bbox.astype(int)

            img_crop = cropImage(frame, xmin, ymin, xmin+width, ymin+height)
            if img_crop.shape[0]*img_crop.shape[1] > 0:
                #Pre-processing the detected image
                # Grayscale
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                # Fixed-height
                img_crop, _ = scaleImgRes(img_crop, height=128)
                # Remove distorted shadow-like pixels. Makes the image highly saturated
                img_crop = filter__removeShadow(img_crop)
                # Unsharp-mask to make image a bit sharper
                #img_crop = unsharp_mask(img_crop)
                setattr(trk_obj, 'latest_frame', img_crop)
                setattr(trk_obj, 'metric_blur', self.variance_of_laplacian(img_crop))

    @staticmethod
    def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    # === End process realm ===


class AsyncObjectTrackerProcessInterface:
    def __init__(self, mot_params : dict, **kwargs):
        #How often to update the tracker
        update_rate : float = mot_params.pop('update_rate', 10.0)
        self._track_only_classes = [1]

        #Load tracker
        _tracker_type = mot_params.pop('type')
        #Get class of the tracker we want to use from the motrackers module
        tracker_cls : typing.Type[Tracker] = getattr(motrackers, _tracker_type) #Throws AttributeError if invalid
        if not issubclass(tracker_cls, Tracker):
            raise ValueError("Tracker type '%s' is not a valid tracker!" % str(_tracker_type))

        self._stopEv : amp.Event = amp.AioEvent(context=mp.get_context())
        self._readyEv : amp.Event = amp.AioEvent(context=mp.get_context())
        self._detections : amp.Queue[typing.Tuple[Detections, np.ndarray, float]] = amp.AioQueue(maxsize=10, context=mp.get_context())
        self._detectionQueue : amp.Queue[typing.OrderedDict] = amp.AioQueue(maxsize = 0, context=mp.get_context())

        self._proc = ObjectTrackerProcess(
            self._stopEv, self._readyEv, self._detections, self._detectionQueue,
            update_rate,
            tracker_cls,
            mot_params
        )

    #Async interface
    async def start(self):
        await asyncio.get_event_loop().run_in_executor(None, self._proc.start)
        with suppress(asyncio.CancelledError):
            await self._readyEv.coro_wait(timeout=3.0)

    async def safeShutdown(self, timeout : float = 10):
        self._stopEv.set()

        # N.B. Not using AioProcess as it is very difficult to subclass it, so manually perform shutdown
        await asyncio.get_event_loop().run_in_executor(None, self._proc.join, timeout)
        if self._proc.is_alive():
            await asyncio.get_event_loop().run_in_executor(None, self._proc.terminate)
        await asyncio.get_event_loop().run_in_executor(None, self._proc.close)

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

        self.tracker = AsyncObjectTrackerProcessInterface(self.tracker_params)
        self.logger.info("Started Tracker task")
        
    async def start_task(self):
        await self.tracker.start()
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
