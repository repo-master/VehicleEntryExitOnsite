
import typing
from vehiclebot.task import AIOTask
import asyncio
import threading
import cv2
import time
import numpy as np

class VideoDisplayDebug(threading.Thread):
    def __init__(self,
                **kwargs):
        super().__init__(daemon=True)

        self._update_rate = 30.0
        self._vehicles = []
        self._frame = np.zeros((100,100,3))
        self._det = None
        self._stopEv = threading.Event()

    #==== Thread realm ====

    def run(self):
        next_time = time.time()
        delaySleep = 0
        while not self._stopEv.is_set():
            frame = self._frame.copy()
            vehicles = self._vehicles.copy()
            for veh in vehicles:
                if veh.is_active:
                    for gate_obj in veh.gate_data.values():
                        (x1, y1, x2, y2, thickness, idx) = gate_obj['gate']
                        x1 *= frame.shape[1]
                        x2 *= frame.shape[1]
                        y1 *= frame.shape[0]
                        y2 *= frame.shape[0]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (128,128,128), thickness)

                    if veh.centroid is not None:
                        angle = veh.movement_direction
                        pos = veh.centroid * frame.shape[1::-1]
                        angle_vec = np.array([np.cos(angle), np.sin(angle)])
                        _pt1 = pos + angle_vec * 64
                        _pt2 = pos - angle_vec * 64
                        cv2.arrowedLine(frame, _pt1.astype(int), _pt2.astype(int), (0,255,128),5)

                trk = veh.associated_track
                if trk is not None:
                    scale = trk._scale
                    inv_scale = 1/scale
                    pt1 = (trk.bbox[0:2]*inv_scale).astype(int)
                    pt2 = (pt1+trk.bbox[2:]*inv_scale).astype(int)
                    cv2.rectangle(frame, pt1, pt2, (0,255,128),3)
                    plate_no = str(veh.license_plate['plate_str']) if veh.license_plate.plate_known else '---'
                    pt_txt = pt1 - (0,12)
                    cv2.putText(frame, plate_no, pt_txt, cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 4)
                    cv2.putText(frame, plate_no, pt_txt, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,255,255), 2)

            scale = 0.3
            h, w = frame.shape[:2]
            fr = cv2.resize(frame, (int(w*scale), int(h*scale)))
            cv2.imshow("camera", fr)
            next_time += (1.0 / self._update_rate)
            delaySleep = next_time - time.time()
            cv2.waitKey(max(1,int(delaySleep*1000)))
        cv2.destroyAllWindows()

    # === End thread realm ===

    async def safeShutdown(self, timeout : float = 10):
        self._stopEv.set()
        await asyncio.get_event_loop().run_in_executor(None, self.join, timeout)


class VideoComposer(AIOTask):
    def __init__(self, tm, task_name,
                 **kwargs):
        super().__init__(tm, task_name, **kwargs)
        self.videoLayers : typing.Dict[int, str] = dict()
        self.vdebug = VideoDisplayDebug()

    async def start_task(self):
        await asyncio.get_event_loop().run_in_executor(None, self.vdebug.start)

    async def stop_task(self):
        await self.vdebug.safeShutdown()

    async def __call__(self):
        self.on('vehicle', self.setVehicles)
        self.on('detect', self.setDetections)

        try:
            capTask = self.tm['camera_source']
            capTask.on('frame', self.setFrame)
        except KeyError:
            pass

    async def setVehicles(self, vehicles):
        self.vdebug._vehicles = vehicles
    async def setFrame(self, frame):
        self.vdebug._frame = frame
    async def setDetections(self, det):
        self.vdebug._det = det

    async def setVideoLayerInputSource(self, layer : int, listen_event : str):
        pass
