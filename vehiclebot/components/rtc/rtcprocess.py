
from ..processing import DetectionHandler
from vehiclebot.management.rtc import DataChannelHandler

import asyncio
import typing
import random
import datetime

class Detection(dict):
    def __init__(self, initial_detection : dict, *args, **kwargs):
        super().__init__(initial_detection, *args, **kwargs)
        self.__initial = initial_detection
    def __call__(self, updated_detection : dict, callback : typing.Callable = None):
        ret_events = []
        if 'is_moving' in updated_detection:
            if not self['is_moving'] and updated_detection['is_moving'] and updated_detection['estimated_movement_direction'] is not 'Stopped':
                #Started moving
                ret_msg = {
                    "title": "Vehicle update",
                    "message": "Vehicle %d started moving towards %s" % (self['track_id'], updated_detection['estimated_movement_direction']),
                    "ts": updated_detection['last_update_ts']
                }
                ret_events.append(ret_msg)
            elif self['is_moving'] and not updated_detection['is_moving']:
                #Stopped moving
                ret_msg = {
                    "title": "Vehicle update",
                    "message": "Vehicle %d stopped moving" % self['track_id'],
                    "ts": updated_detection['last_update_ts']
                }
                ret_events.append(ret_msg)

        if 'plate' in updated_detection and 'plate' not in self:
            '''ret_msg = {
                "title": "Vehicle update",
                "message": "Vehicle %d recognised as \"%s\"" % (self['track_id'], updated_detection['plate']['plate_number']),
                "ts": updated_detection['plate']['detect_ts']
            }
            ret_events.append(ret_msg)'''
            pass

        self.update(updated_detection)

        if len(ret_events) > 0:
            return ret_events

class RTCDataProcess(DetectionHandler):
    def __init__(self, *args, plate_decoder = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.handlers : typing.Dict[str, DataChannelHandler] = dict(
            cmd = DataChannelHandler(),
            log = DataChannelHandler(),
            status = DataChannelHandler()
        )
        self.plate_decoder = plate_decoder
        self._all_detections : typing.Dict[int, Detection] = {}
        self._new_client_messages = []
        
    async def start_task(self):
        await super().start_task()
        self.tm.app['rtc'].addInDataChannelHandler(**self.handlers)
        self.handlers['log'].on('data', self._handleMessageLog)
        self.handlers['log'].on('connect', self._handleLogSendAll)

    async def __call__(self):
        while not self._stop.is_set():
            is_stop, next_items = await asyncio.gather(self._stop.wait_for(timeout=.1), self.save_queue.get_wait_for(timeout=.3))
            if is_stop: break
            if next_items is None: continue
            
            for batch in next_items:
                for det in batch:
                    if det['is_new_detection']:
                        new_det = Detection(det)
                        self._submitPlateDecodeTask(new_det)

                        self._all_detections[det['track_id']] = new_det
                        msg = {
                            "title": "Vehicle detected",
                            "message": "New vehicle detected, recognising plate...",
                            "ts": new_det['first_detect_ts']
                        }
                        self.handlers['log'].broadcast(msg)
                        self._new_client_messages.append(msg)
                    else:
                        dtect : Detection = self._all_detections.get(det['track_id'])
                        if dtect is not None:
                            event = dtect(det) #Update with new detection frame
                            if event:
                                self.handlers['log'].broadcast(event)
                                self._new_client_messages.extend(event)
            
            #Task progress check (periodically checked)
            for det in self._all_detections.values():
                self._checkPlateTask(det)
                
            self.save_queue.task_done()

            #Not efficient but for now its okay. Later needs to use delta and send on update
            await self._updateVehicleData()
            
    def _submitPlateDecodeTask(self, det : Detection):
        if self.plate_decoder is None: return
        setattr(det, '_rec_task', asyncio.create_task(
            self.tm[self.plate_decoder].detectAndDecode(dict(det))
        ))

    def _checkPlateTask(self, det : Detection):
        rec_task : asyncio.Task = getattr(det, '_rec_task', None)
        if rec_task is not None:
            if rec_task.done():
                plate_detection = rec_task.result()
                #Bad detection, maybe try again
                if plate_detection is None:
                    #TODO: Bit delay in frames before retrying
                    #self._submitPlateDecodeTask(det)
                    return

                if plate_detection.get('code') < 0:
                    #self.logger.debug("Plate number was not determined: %s", plate_detection.get('message'))
                    return
                
                event = det({"plate": plate_detection})
                if event:
                    self.handlers['log'].broadcast(event)
                    self._new_client_messages.extend(event)
                delattr(det, '_rec_task')

    async def _handleMessageLog(self, msg, channel, peer):
        self.logger.info("Got message at channel %s: %s" % (channel.label, msg))

    async def _handleLogSendAll(self, channel):
        channel.send(self._new_client_messages)

    async def _updateVehicleData(self):
        self.handlers['status'].broadcast(sorted([
            {
                "track_id": det['track_id'],
                "plate_number": (det['plate']['plate_str'] or 'Unknown') if 'plate' in det else 'Detecting...',
                "detect_time": det.get('first_detect_ts')
            }
            for det in self._all_detections.values()
        ], key=lambda o: o['detect_time'], reverse=True))
