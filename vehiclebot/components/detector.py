
import datetime
from vehiclebot.task import AIOTask
from vehiclebot.multitask import AsyncProcess

from vehiclebot.model import YoloModelCV2
from vehiclebot.tracker import Trajectory, TRACK_COLORS, Tracker, motrackers

import os
import cv2
import asyncio
import typing

#Synchronous code in isolated process
class VehicleDetectorProcess:
    def __init__(self, model_file : os.PathLike, tracker_config : dict, output_config : dict):
        self._last_detection = None
        self._last_img = None
        self.traj = Trajectory()
        self.opcfg = output_config

        #Load model file directly
        self.model = YoloModelCV2.fromZip(model_file) #Throws FileNotFoundError if model is not found
        
        _tracker_type = tracker_config.pop('type')
        tracker_cls : typing.Type[Tracker] = getattr(motrackers, _tracker_type) #Throws AttributeError if invalid
        if not issubclass(tracker_cls, Tracker):
            raise ValueError("Tracker type \"%s\" is not a valid tracker!" % str(_tracker_type))
        
        self.tracker = tracker_cls(
            tracker_output_format='mot_challenge',
            **tracker_config
        )

    def detect(self, img):
        self._last_detection = self.model.detect(img, zip_results=False, label_str=False, min_score=0.2)
        self._last_img = img
        
    def updateTracker(self):
        output_tracks = self.tracker.update(*self._last_detection)
        self._updateUnmatchedTracks()
        for trk in output_tracks:
            trk_id = trk[1]
            trk_obj = self.tracker.tracks[trk_id]
            #Timestamp
            if not hasattr(trk_obj, 'detect_ts'):
                setattr(trk_obj, 'detect_ts', datetime.datetime.now(datetime.timezone.utc))
            setattr(trk_obj, 'update_ts', datetime.datetime.now(datetime.timezone.utc))

            xmin, ymin, width, height = trk[2:6]
            xcentroid, ycentroid = xmin + 0.5*width, ymin + 0.5*height
            #Update trajectory of this track
            self.traj.update(trk_id, (xcentroid, ycentroid))

    def _updateUnmatchedTracks(self):
        output_tracks = self.tracker.tracks
        for trk_id, trk_obj in output_tracks.items():
            if trk_obj.lost:
                pass
                #print(trk_id, "is lost", flush=True)

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
        
            txt_str = "{confidence:.2f}%".format(confidence=conf*100)
            trk_text = "ID {}".format(trk_id) + (' LOST' if trk_obj.lost else '')
            
            ##Drawing
            #Bounding box
            cv2.rectangle(img, (xmin, ymin), (xmin+width, ymin+height), trk_color, 1 if trk_obj.lost else 2)
        
            #Class and confidence
            txt_pos = [xmin, ymin]
            txt_pos[0] = max(txt_pos[0], 2)
            txt_pos[1] = max(txt_pos[1]-8, 10)
            cv2.putText(img, txt_str, txt_pos, cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
            cv2.putText(img, txt_str, txt_pos, cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 255, 255), lineType=cv2.LINE_AA, thickness=1)
            #Track id
            cv2.putText(img, trk_text, (xcentroid - 10, ycentroid - height//2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.5 if trk_obj.lost else 1.2, (0, 0, 0), lineType=cv2.LINE_AA, thickness=4)
            cv2.putText(img, trk_text, (xcentroid - 10, ycentroid - height//2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.5 if trk_obj.lost else 1.2, trk_color, lineType=cv2.LINE_AA, thickness=2)
            #Track center
            cv2.circle(img, (xcentroid, ycentroid), 4, trk_color, -1)
            
    def drawTracks(self, img=None):
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

    def cropAndProcessTracks(self, img=None):
        results = []
        if img is None:
            if self._last_img is None: return results
            img = self._last_img.copy()
        
        output_tracks = self.tracker.tracks
        for _, trk_obj in output_tracks.items():
            trk = trk_obj.output()
            
            trk_id = trk[1]
            xmin, ymin, width, height = trk[2:6]
            conf = float(trk[6])
            cls_lbl = trk_obj.class_id
            cls_lbl = self.model.getLabel(cls_lbl)
            detect_ts, update_ts = None, None
            if hasattr(trk_obj, 'detect_ts'):
                detect_ts = trk_obj.detect_ts
            if hasattr(trk_obj, 'update_ts'):
                update_ts = trk_obj.update_ts
            traj_dir_angle = self.traj[trk_id]['dir']
            traj_dir_mv = self.traj[trk_id]['cardinal']
            traj_mv_spd = self.traj[trk_id]['speed']
            traj_is_mv = self.traj[trk_id]['is_moving']
            
            #Integer to crop
            xmin, ymin, width, height = int(xmin), int(ymin), int(width), int(height)
            
            if not trk_obj.lost and width > 0 and height > 0:
                encode_format = self.opcfg['image_format']
                img_crop = self._crop(img, xmin, ymin, xmin+width, ymin+height)
                if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                    ret, img_blob = cv2.imencode(encode_format, img_crop)
                    if ret:
                        res_item = {
                            'track_id': trk_id,
                            'age': trk_obj.age,
                            'img': {
                                'encoding': "encoded_np",
                                'data': img_blob,
                                'format': encode_format
                            },
                            'is_new_detection': trk_obj.age <= 1,
                            'class': cls_lbl,
                            'class_confidence': conf,
                            'bbox': (xmin, ymin, width, height),
                            'first_detect_ts': detect_ts,
                            'last_update_ts': update_ts,
                            'movement_speed': traj_mv_spd,
                            'is_moving': traj_is_mv,
                            'estimated_movement_angle': traj_dir_angle,
                            'estimated_movement_direction': traj_dir_mv
                        }
                        results.append(res_item)

        return results

class VehicleDetector(AIOTask, AsyncProcess):
    metadata : typing.Dict[str, typing.Any] = {"dependencies": []}
    def __init__(self, tm, task_name,
                 input_source,
                 detector : typing.Dict,
                 tracker: typing.Dict,
                 output : typing.Dict = None,
                 video_output : str = None,
                 output_result : str = None,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.inp_src = input_source
        self.detector_params = detector
        self.tracker_params = tracker
        self.video_output = video_output
        self.output_result = output_result
        self.output_config = output
        if self.output_config is None:
            self.output_config = {"image_format": ".png"}

        self.detector : VehicleDetectorProcess = None
        self._stop = asyncio.Event()
        self.prepareProcess()
        self.logger.info("Started Detector worker process")
        
    async def start_task(self):
        try:
            self.logger.info("Loading model \"%s\" for detector, and tracker using \"%s\"..." % (self.detector_params['model'], self.tracker_params['type']))
            self.detector = await self.asyncCreate(VehicleDetectorProcess, self.detector_params['model'], self.tracker_params, self.output_config)
        except (FileNotFoundError, AttributeError, ValueError):
            self.logger.exception("Error loading detection model")
            
    async def stop_task(self):
        self._stop.set()
        await self.task
        self.endProcess()

    async def __call__(self):
        try:
            cap = self.tm[self.inp_src]
        except KeyError:
            self.logger.error("Input source component \"%s\" is not loaded. Please check your config file to make sure it is properly configured." % self.inp_src)
            return
        
        if self.output_result is not None:
            if self.output_result not in self.tm.tasks:
                self.logger.error("Required task \"%s\" is not loaded" % self.output_result)
                return
            
            #TODO
            # if not isinstance(self.tm[self.output_result], DetectionHandler):

        '''
        Operation:
          - Grab frames as fast as possible (or as necessary)
          - Perform detections and
          - Update the tracker with detected objects
        '''
        while not await self._stop.wait_for(0.1):
            #Try to grab a frame from video source
            img = cap.frame
            if img is not None:
                # Perform detection using the DNN model
                await self.detector.detect(img)
                # and then perform object tracking
                await self.detector.updateTracker()

                # Show video output
                proc_vout = self.render_results()
                # Send results to processing stage
                # Need to pass the image as this process is 'fire-and-forget', hence not re-entrant
                proc_push = self.push_results(img)
                
                # Wait till both tasks are done
                await asyncio.gather(proc_vout, proc_push)

    async def render_results(self):
        if self.video_output is not None:
            #Draw tracks on the frame
            img_drawn = await self.detector.drawTracks()
            if img_drawn is None: return
            vidisplay = self.tm[self.video_output]
            await vidisplay.imshow("Detections", img_drawn)

    async def push_results(self, img):
        if self.output_result is not None:
            detections_data_task = self.detector.cropAndProcessTracks(img)
            await self.tm[self.output_result].processDetection(detections_data_task)
