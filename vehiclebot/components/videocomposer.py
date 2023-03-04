
import typing
from vehiclebot.task import AIOTask
import asyncio
import threading
import cv2
import time
import numpy as np
from datetime import datetime
import queue
from contextlib import suppress

class VideoDisplayDebug(threading.Thread):
    def __init__(self,
                **kwargs):
        self.record = False
        self.renderToWindow = False

        self._update_rate = 30.0
        self._vehicles = []
        self._ee_data = tuple()
        self._frame = self._noSignalImage()
        self._det = None
        self._stopEv = threading.Event()
        # Frame queue to render and store for asyncio thread to consume
        self._frameQ = asyncio.Queue(maxsize=8)

        if self.record:
            self._wr = cv2.VideoWriter("test_results/video_%s.mp4" % datetime.now().strftime("%Y%m%d_%H%M"), cv2.VideoWriter_fourcc(*'MP4V'), 50, (1280, 720))

        super().__init__(daemon=True)

    def _noSignalImage(self):
        '''Simply creates an image with "No signal" written'''
        img = np.zeros((360,640,3))
        no_signal = {
            'text': "No signal",
            'fontFace': cv2.FONT_HERSHEY_COMPLEX,
            'fontScale': 1.2,
            'thickness': 5
        }

        text_width, text_height = cv2.getTextSize(**no_signal)[0]
        CenterCoordinates = (int(img.shape[1] / 2)-int(text_width / 2), int(img.shape[0] / 2) - int(text_height / 2))

        cv2.putText(img, **no_signal, org=CenterCoordinates, color=(255,255,255))
        no_signal['thickness'] = 2
        cv2.putText(img, **no_signal, org=CenterCoordinates, color=(0,0,255))
        return img

    #==== Thread realm ====

    def run(self):
        next_time = time.time()
        delaySleep = 0
        while not self._stopEv.is_set():
            frame = self._frame.copy()
            vehicles = self._vehicles.copy()
            inv_scale = np.array(frame.shape[1::-1])

            if len(self._ee_data) == 2:
                ee_veh, ee_log = self._ee_data

                for i, v in enumerate(ee_veh):
                    a = "Type: {type}, Plate: {plate_number}, State: {entry_exit_state} at {x}".format(**v,
                    x = v['last_state_timestamp'].strftime('%Y-%m-%d %H:%M:%S') if v['last_state_timestamp'] is not None else '-')
                    pt_txt = (60,60+i*40)
                    cv2.putText(frame, a, pt_txt, cv2.FONT_HERSHEY_COMPLEX, 1.1, (0,0,0), 4)
                    cv2.putText(frame, a, pt_txt, cv2.FONT_HERSHEY_COMPLEX, 1.1, (255,255,255), 2)

            for veh in vehicles:
                if veh.is_active:
                    for gate_obj in veh.gate_data.values():
                        (x1, y1, x2, y2), normal = gate_obj['coords'], gate_obj['normal']
                        x1 *= frame.shape[1]
                        x2 *= frame.shape[1]
                        y1 *= frame.shape[0]
                        y2 *= frame.shape[0]
                        center = ((x1+x2)/2, (y1+y2)/2)
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (128,128,128), 5)
                        #Normal vector
                        angle_vec = np.array([np.cos(normal), -np.sin(normal)])
                        _pt1 = center - angle_vec * 64
                        _pt2 = center + angle_vec * 64
                        cv2.arrowedLine(frame, _pt1.astype(int), _pt2.astype(int), (128,128,255),3)

                    if veh.centroid is not None:
                        pos = veh.centroid * inv_scale
                        if veh.is_moving:
                            angle_vec = veh.velocity
                            _pt1 = pos
                            _pt2 = pos + angle_vec * 96
                            cv2.arrowedLine(frame, _pt1.astype(int), _pt2.astype(int), (0,255,128),4)
                        else:
                            cv2.circle(frame, pos.astype(int), 10, (0,0,128), -1)
                            cv2.circle(frame, pos.astype(int), 10, (255,255,255), 1)

                trk = veh.associated_track
                if trk is not None:
                    pt1 = (trk.bbox[0:2]*inv_scale).astype(int)
                    pt2 = (pt1+trk.bbox[2:]*inv_scale).astype(int)
                    cv2.rectangle(frame, pt1, pt2, (128,96,96), 2)
                    plate_no = str(veh.license_plate['plate_str']) if veh.license_plate.plate_known else '---'
                    pt_txt = pt1 - (0,12)
                    cv2.putText(frame, plate_no, pt_txt, cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 4)
                    cv2.putText(frame, plate_no, pt_txt, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,255,255), 2)

            #All detections
            if self._det is not None:
                for det, score, cls in zip(*self._det):
                    pt1 = (det[:2] * inv_scale).astype(int)
                    pt2 = (pt1 + det[2:4] * inv_scale).astype(int)
                    self.draw_border(frame, pt1, pt2, (0,255,255),3)

            #Save rendered frame to the frame queue
            with suppress(asyncio.QueueFull):
                self._frameQ.put_nowait(frame.astype(np.uint8))

            if self.record:
                self._wr.write(cv2.resize(frame, (1280,720)))

            #Render to screen in a window
            if self.renderToWindow:
                scale = 0.6
                h, w = frame.shape[:2]
                fr = cv2.resize(frame, (int(w*scale), int(h*scale)))
                cv2.imshow("camera", fr)

            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            cv2.waitKey(max(1,int(delaySleep*1000)))

        cv2.destroyAllWindows()
        if self.record:
            self._wr.release()

    @staticmethod
    def draw_border(img, pt1, pt2, color, thickness, r = 10, d = 20):
        '''Draw fancy rounded rectangle corners'''
        x1,y1 = pt1
        x2,y2 = pt2

        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # === End thread realm ===

    async def safeShutdown(self, timeout : float = 10):
        self._stopEv.set()
        await asyncio.get_event_loop().run_in_executor(None, self.join, timeout)

    async def videoFrameUpdated(self):
        while not self._stopEv.is_set():
            with suppress(queue.Empty):
                yield await self._frameQ.get_wait_for(timeout=1.0)

class VideoComposer(AIOTask):
    def __init__(self, tm, task_name,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.videoLayers : typing.Dict[int, str] = dict()
        self._cTask = None
        self.vdebug = VideoDisplayDebug()

    async def start_task(self):
        await asyncio.get_event_loop().run_in_executor(None, self.vdebug.start)

    async def stop_task(self):
        await self.vdebug.safeShutdown()
        if self._cTask is not None:
            with suppress(asyncio.CancelledError):
                self._cTask.cancel()
                await self._cTask

    async def __call__(self):
        self.on('frameIn', self.setFrame)
        self.on('vehicle', self.setVehicles)
        self.on('detect', self.setDetections)

        self._cTask = asyncio.create_task(self.emitFrames())

    async def emitFrames(self):
        async for frame in self.vdebug.videoFrameUpdated():
            self.emit('frame', frame)

    async def setVehicles(self, vehicles):
        self.vdebug._vehicles = vehicles
        self.vdebug._ee_data = self.tm['vehicledata'].generateDashboardData()
    async def setFrame(self, frame):
        self.vdebug._frame = frame
    async def setDetections(self, detection, frame, scale):
        self.vdebug._det = detection

    async def setVideoLayerInputSource(self, layer : int, listen_event : str):
        pass
