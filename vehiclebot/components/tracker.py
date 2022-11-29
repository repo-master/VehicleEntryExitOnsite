
from ..task import AIOTask, TaskOrTasks
from ..multitask import AsyncProcess

from .camera import CameraSource, scaleImgRes
from .detector import Detection

from ..tracker import Trajectory, Tracker, motrackers, TRACK_COLORS

import cv2
import time
import queue
import typing
import asyncio
import threading

import numpy as np

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

    def updateDetection(self, det_frame : np.ndarray):
        '''Update tracking features from the frame to use as template'''
        if det_frame is None:
            return
        
        #Image returned by detector is already scaled according to detect resolution

        for trk_obj in self._tracker.tracks.values():
            _tracker_id = self.trackerID()
            if not hasattr(trk_obj, _tracker_id):
                setattr(trk_obj, _tracker_id, self._trackfact.create())
                
            setattr(trk_obj, "%s_bbox" % _tracker_id, tuple(trk_obj.bbox))
            setattr(trk_obj, '_is_lost', trk_obj.lost > 20)

            is_fresh = trk_obj.lost == 0

            if is_fresh:
                #Init feature tracker with new frame
                ftrack = getattr(trk_obj, _tracker_id)
                ftrack.init(det_frame, trk_obj.bbox.astype(int))
                
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
                success, new_bbox = ftrack.update(frame_scaled)
                if success:
                    setattr(trk_obj, "%s_bbox" % _tracker_id, new_bbox)
                   
                needs_tracker_nudge = (
                    trk_obj._is_lost or
                    (time.time()-trk_obj._update_ts) > 2.0
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
        
        self._feature_tracker = FeatureTracker(self.tracker)
        self._trajectory = Trajectory(self.tracker)

        self._stopEv = threading.Event()
        self._detections : queue.Queue[typing.Tuple[Detection, np.ndarray, float]] = queue.Queue(maxsize=10)

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
            self._feature_tracker()
            self._trajectory(self._feature_tracker.getBBox)
            
            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            if delaySleep < 0:
                delaySleep = 0
                next_time = time.time()

        self.cleanup()

    def _updateTracker(self, dets : Detection, frame : np.ndarray, scale : float):
        self.tracker.update(*dets)
        self._updateTrackerStats(scale)
        self._feature_tracker.updateDetection(frame)
        
    def _updateTrackerStats(self, scale : float):
        setattr(self.tracker, '_scale', scale)
        for trk_obj in self.tracker.tracks.values():
            setattr(trk_obj, '_update_ts', time.time())
            if trk_obj.age == 1:
                setattr(trk_obj, '_create_ts', trk_obj._update_ts)
                

    #Public methods

    def updateVideoFrame(self, frame : np.ndarray):
        '''Update feature tracker with new video frame'''
        self._feature_tracker.updateTrackFrame(frame)
        
    def updateDetection(self, detection : Detection, frame : np.ndarray, scale : float):
        '''Add new detection to queue for processing'''
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
            trk_text = "ID {}".format(trk_id) + (' LOST' if is_lost else '')
            
            ##Drawing
            #Bounding box
            if ft_box is not None:
                cv2.rectangle(img, *ft_box, (100,200,200), 1)

            cv2.rectangle(img, (xmin, ymin), (xmin+width, ymin+height), trk_color, 1 if is_lost else 2)
        
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

class ObjectTracker(AIOTask, AsyncProcess):
    def __init__(self, tm, task_name,
                 input_source : str,
                 tracker: typing.Dict,
                 output : TaskOrTasks = None,
                 update_rate : float = 10.0,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.inp_src = input_source
        self.output_dest = output
        self._update_rate = update_rate
        self.tracker_params = tracker

        self.tracker : TrackerProcess = None
        self._stopEv = asyncio.Event()
        self.logger.info("Started Tracker task")
        
    async def start_task(self):
        await self._initEvents()
        await self.prepareProcess()
        try:
            self.tracker = await self.asyncCreate(TrackerProcess, self.tracker_params)
        except (FileNotFoundError, AttributeError, ValueError):
            self.logger.exception("Error loading detection model")
            
    async def stop_task(self):
        self._stopEv.set()
        await self.task

    async def __call__(self):
        try:
            capTask : CameraSource = self.tm[self.inp_src]
        except KeyError:
            self.logger.error("Input source component \"%s\" is not loaded. Please check your config file to make sure it is properly configured." % self.inp_src)
            return
        
        if self._update_rate is None:
            self._update_rate = 60

        next_time = time.time()
        delaySleep = 0
        #Update tracker frame at a steady rate
        while not await self._stopEv.wait_for(delaySleep):
            img = await capTask.frame()
            if img is not None:
                await self.tracker.updateVideoFrame(img)
                img = await self.tracker.drawTrackBBoxes(img)
                await self.tm.emit(self.output_dest, "frame", "Tracker", img)
            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            if delaySleep < 0:
                delaySleep = 0
                next_time = time.time()

        await self.tracker.stop()

    #Event handlers
    async def _initEvents(self):
        self.on('detect', self.updateDetection)
        

    #Input methods
    async def updateDetection(self, detection : Detection, frame : np.ndarray, scale : float = 1.0):
        '''Update the tracker with new set of detections'''
        if self.tracker is None: return
        await self.tracker.updateDetection(detection, frame, scale)
