
from vehiclebot.task import AIOTask
from vehiclebot.multitask import GlobalInstances

from vehiclebot.model import YOLOModel
from vehiclebot.tracker import SORT, Trajectory, TRACK_COLORS

import cv2
import asyncio
import typing

from concurrent.futures import ProcessPoolExecutor

#Synchronous code in isolated process
class VehicleDetectorProcess(GlobalInstances):
    def __init__(self, model_file):
        self._last_detection = None
        self.model = YOLOModel.fromZip(model_file)
        self.traj = Trajectory()
        self.tracker = SORT(
            max_lost=8,
            tracker_output_format='mot_challenge',
            iou_threshold=0.6
        )

    def _detect(self, img):
        self._last_detection = self.model.detect(img, zip_results=False, label_str = False)
        
    def _updateTracker(self):
        self.tracker.update(*self._last_detection)
        output_tracks = self.tracker.tracks #So that we can get all tracks instead of active ones
        
        for _, trk_obj in output_tracks.items():
            trk = trk_obj.output()
            trk_id = trk[1]
            xmin, ymin, width, height = trk[2:6]
            xcentroid, ycentroid = xmin + 0.5*width, ymin + 0.5*height
            #Update trajectory
            if trk_obj.lost == 0:
                self.traj.update(trk_id, (xcentroid, ycentroid))
    
    def _drawTrackBBoxes(self, img):
        output_tracks = self.tracker.tracks
        for _, trk_obj in output_tracks.items():
            trk = trk_obj.output()
            
            #Extract all information to display
            trk_id = trk[1]
            xmin, ymin, width, height = trk[2:6]
            trk_color = TRACK_COLORS[
                -1 if trk_obj.lost else
                (trk_id % (len(TRACK_COLORS)-1))
            ]
            conf = trk[6]
            cls_lbl = trk_obj.class_id
            cls_lbl = self.model.getLabel(cls_lbl)
            
            #Calculate centroid of current detection
            xcentroid, ycentroid = xmin + 0.5*width, ymin + 0.5*height

            #Integer for drawing
            xmin, ymin, width, height = int(xmin), int(ymin), int(width), int(height)
            xcentroid, ycentroid = int(xcentroid), int(ycentroid)
        
            txt_str = "Class: {label} ({conf:.2f}%)".format(label=cls_lbl, conf=conf*100)
            trk_text = "ID {}".format(trk_id) + (' LOST' if trk_obj.lost else '')
            
            ##Drawing
            #Bounding box
            cv2.rectangle(img, (xmin, ymin), (xmin+width, ymin+height), trk_color, 1 if trk_obj.lost else 2)
        
            #Class and confidence
            txt_pos = [xmin, ymin]
            txt_pos[0] = max(txt_pos[0], 2)
            txt_pos[1] = max(txt_pos[1]-8, 10)
            cv2.putText(img, txt_str, txt_pos, cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
            cv2.putText(img, txt_str, txt_pos, cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 255, 255), lineType=cv2.LINE_AA, thickness=1)
            #Track id
            cv2.putText(img, trk_text, (xcentroid - 10, ycentroid - height//2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3 if trk_obj.lost else 0.5, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
            cv2.putText(img, trk_text, (xcentroid - 10, ycentroid - height//2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3 if trk_obj.lost else 0.5, trk_color, lineType=cv2.LINE_AA, thickness=1)
            #Track center
            cv2.circle(img, (xcentroid, ycentroid), 4, trk_color, -1)
            
    def _drawTracks(self, img):
        self._drawTrackBBoxes(img)
        self.traj.draw(img)
        return img

    @classmethod
    def instantiate(cls, model_file):
        return cls.create_instance(cls(model_file))
    
    @classmethod
    def detect(cls, instance, img):
        return cls.get_instance(instance)._detect(img)
    
    @classmethod
    def updateTracker(cls, instance):
        return cls.get_instance(instance)._updateTracker()
        
    @classmethod
    def drawTracks(cls, instance, img):
        return cls.get_instance(instance)._drawTracks(img)


class VehicleDetector(AIOTask):
    def __init__(self, tm, task_name, input_source, detector : typing.Dict, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.inp_src = input_source
        self.detector_params = detector
        self.detector = None
        self._stop = False
        self.proc = ProcessPoolExecutor(max_workers=1, initializer=VehicleDetectorProcess.init)

    def call_process(self, func, *args):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.proc, func, *args)
    
    async def start_task(self):
        try:
            self.logger.info("Loading model \"%s\" for detector..." % self.detector_params['model'])
            self.detector = await self.call_process(VehicleDetectorProcess.instantiate, self.detector_params['model'])
        except FileNotFoundError as e:
            self.logger.exception(e)

    async def stop_task(self):
        self._stop = True
        await self.task

    async def __call__(self):
        '''
        Operation:
          - Grab frames as fast as possible (or as necessary)
          - Perform detections and
          - Update the tracker with detected objects
        '''
        while not self._stop:
            cap = self.tm[self.inp_src]
            img = cap.frame
            if img is not None:
                #img_copy = img.copy()

                # Perform detection using the DNN model
                await self.call_process(VehicleDetectorProcess.detect, self.detector, img)
                # and then perform object tracking
                await self.call_process(VehicleDetectorProcess.updateTracker, self.detector)
                
                #Draw tracks on the frame
                img_drawn = await self.call_process(VehicleDetectorProcess.drawTracks, self.detector, img)

                await self.tm['videodisplay'].imshow("Detections", img_drawn)
            
            await asyncio.sleep(0.1)
