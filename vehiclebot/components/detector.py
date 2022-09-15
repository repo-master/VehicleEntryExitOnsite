
import datetime
from vehiclebot.task import AIOTask
from vehiclebot.multitask import GlobalInstances

from vehiclebot.model import YOLOModel
from vehiclebot.tracker import Trajectory, TRACK_COLORS, Tracker, motrackers

import os
import cv2
import asyncio
import typing

from concurrent.futures import ProcessPoolExecutor

#Synchronous code in isolated process
class VehicleDetectorProcess(GlobalInstances):
    def __init__(self, model_file : os.PathLike, tracker_opts : typing.Dict):
        self._last_detection = None
        self._last_img = None
        self.traj = Trajectory()

        #Load model file directly
        self.model = YOLOModel.fromZip(model_file) #Throws FileNotFoundError if model is not found
        
        _tracker_type = tracker_opts.pop('type')
        tracker_cls : typing.Type[Tracker] = getattr(motrackers, _tracker_type) #Throws AttributeError if invalid
        if not issubclass(tracker_cls, Tracker):
            raise ValueError("Tracker type \"%s\" is not a valid tracker!" % str(_tracker_type))
        
        self.tracker = tracker_cls(
            tracker_output_format='mot_challenge',
            **tracker_opts
        )

    def _detect(self, img, cache_results=True):
        self._last_detection = self.model.detect(img, zip_results=False, label_str=False)
        if cache_results: self._last_img = img
        
    def _updateTracker(self):
        output_tracks = self.tracker.update(*self._last_detection)        
        for trk in output_tracks:
            trk_id = trk[1]
            trk_obj = self.tracker.tracks[trk_id]
            #Timestamp
            if not hasattr(trk_obj, 'detect_ts'):
                setattr(trk_obj, 'detect_ts', datetime.datetime.now(datetime.timezone.utc))

            xmin, ymin, width, height = trk[2:6]
            xcentroid, ycentroid = xmin + 0.5*width, ymin + 0.5*height
            #Update trajectory of this track
            self.traj.update(trk_id, (xcentroid, ycentroid))
    
    def _drawTrackBBoxes(self, img):
        output_tracks = self.tracker.tracks #So that we can get all tracks instead of active ones
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
            
    def _drawTracks(self, img=None):
        if img is None:
            if self._last_img is None: return
            img = self._last_img.copy()

        self._drawTrackBBoxes(img)
        self.traj.draw(img)
        return img

    @staticmethod
    def _crop(img, x1, y1, x2, y2):
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = max(x2, 0)
        y2 = max(y2, 0)
        return img[y1:y2,x1:x2]

    def _cropAndProcessTracks(self, img=None):
        results = []
        if img is None:
            if self._last_img is None: return results
            img = self._last_img.copy()
        
        output_tracks = self.tracker.tracks
        for _, trk_obj in output_tracks.items():
            trk = trk_obj.output()
            
            trk_id = trk[1]
            xmin, ymin, width, height = trk[2:6]
            conf = trk[6]
            cls_lbl = trk_obj.class_id
            cls_lbl = self.model.getLabel(cls_lbl)
            detect_ts = None
            if hasattr(trk_obj, 'detect_ts'):
                detect_ts = trk_obj.detect_ts
            traj_dir_angle = self.traj[trk_id]['dir']

            #Integer to crop
            xmin, ymin, width, height = int(xmin), int(ymin), int(width), int(height)
            
            if not trk_obj.lost and width > 0 and height > 0:
                ENCODE_FORMAT = '.png'
                img_crop = self._crop(img, xmin, ymin, xmin+width, ymin+height)
                if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                    ret, img_blob = cv2.imencode(ENCODE_FORMAT, img_crop)
                    if ret:
                        res_item = {
                            'track_id': trk_id,
                            'img': img_blob,
                            'format': ENCODE_FORMAT,
                            'is_new_detection': trk_obj.age <= 1,
                            'class': cls_lbl,
                            'class_confidence': conf,
                            'bbox': tuple(trk[2:6]),
                            'first_detect_ts': detect_ts,
                            'last_update_ts': datetime.datetime.now(datetime.timezone.utc),
                            'estimated_movement_angle': traj_dir_angle,
                            'estimated_movement_direction': ''
                        }
                        results.append(res_item)

        return results

    @classmethod
    def instantiate(cls, model_file, tracker):
        return cls.create_instance(cls(model_file, tracker))
    
    @classmethod
    def detect(cls, instance, img, cache_results=True):
        return cls.get_instance(instance)._detect(img, cache_results)
    
    @classmethod
    def updateTracker(cls, instance):
        return cls.get_instance(instance)._updateTracker()
        
    @classmethod
    def drawTracks(cls, instance, img=None):
        return cls.get_instance(instance)._drawTracks(img)

    @classmethod
    def cropAndProcessTracks(cls, instance, img=None):
        return cls.get_instance(instance)._cropAndProcessTracks(img)


class VehicleDetector(AIOTask):
    metadata : typing.Dict[str, typing.Any] = {"dependencies": []}
    def __init__(self, tm, task_name, input_source, detector : typing.Dict, tracker: typing.Dict, video_output : str = None, output_result : str = None, **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.inp_src = input_source
        self.detector_params = detector
        self.tracker_params = tracker
        self.video_output = video_output
        self.detector = None
        self._stop = False
        self.proc = ProcessPoolExecutor(max_workers=1, initializer=VehicleDetectorProcess.init)

    async def call_process(self, func, *args):
        task = asyncio.get_event_loop().run_in_executor(self.proc, func, *args)
        task.add_done_callback(self._proc_done_callback)
        return await task

    def _proc_done_callback(self, future : asyncio.Future):
        if future.exception():
            self.logger.exception(future.exception())
    
    async def start_task(self):
        try:
            self.logger.info("Loading model \"%s\" for detector, and tracker using \"%s\"..." % (self.detector_params['model'], self.tracker_params['type']))
            self.detector = await self.call_process(VehicleDetectorProcess.instantiate, self.detector_params['model'], self.tracker_params)
        except (FileNotFoundError, AttributeError, ValueError) as e:
            self.logger.exception(e)

    async def stop_task(self):
        self._stop = True
        await self.task

    async def __call__(self):
        try:
            cap = self.tm[self.inp_src]
        except KeyError:
            self.logger.warning("Input source component \"%s\" is not loaded. Please check your config file to make sure it is properly configured." % self.inp_src)
            return

        '''
        Operation:
          - Grab frames as fast as possible (or as necessary)
          - Perform detections and
          - Update the tracker with detected objects
        '''
        while not self._stop:
            #Try to grab a frame from video source
            img = cap.frame
            if img is not None:
                # Perform detection using the DNN model
                await self.call_process(VehicleDetectorProcess.detect, self.detector, img)
                # and then perform object tracking
                await self.call_process(VehicleDetectorProcess.updateTracker, self.detector)

                # Show video output
                proc_vout = self.render_results()
                # Send results to processing stage
                proc_push = self.push_results()
                
                await asyncio.gather(proc_vout, proc_push)
            else:
                #Do not hog asyncio's time from the tight loop
                await asyncio.sleep(0.1)

    async def render_results(self):
        if self.video_output is not None:
            #Draw tracks on the frame
            img_drawn = await self.call_process(VehicleDetectorProcess.drawTracks, self.detector)
            if img_drawn is None: return
            vidisplay = self.tm[self.video_output]
            await vidisplay.imshow("Detections", img_drawn)

    async def push_results(self):
        res = await self.call_process(VehicleDetectorProcess.cropAndProcessTracks, self.detector)
        print(res)
