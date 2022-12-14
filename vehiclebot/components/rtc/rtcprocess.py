
from ..processing import DetectionHandler
from vehiclebot.management.rtc import DataChannelHandler

import asyncio
import typing
import numpy as np
import datetime
import uuid
from time import time

TMP_CLASSID_TO_TYPE : typing.Dict[int, str] = {
    0: 'Car',
    1: 'Car',
    2: 'Rickshaw',
    3: 'Two-wheeler',
    4: 'Car',
    5: 'Car',
    6: 'Car',
    7: 'Car',
    8: 'Car',
    9: 'Car',
    10: 'Two-wheeler',
    11: 'Car',
    12: 'Car',
    13: 'Rickshaw',
    14: 'Two-wheeler',
    15: 'Car',
    16: 'Car',
    17: 'Car',
    18: 'Car',
    19: 'Car',
    20: 'Car'
}

class Detection(dict):
    def __init__(self, initial_detection : dict, *args, **kwargs):
        super().__init__(initial_detection, *args, **kwargs)
        self.__initial = initial_detection
        self['track_uuid'] = uuid.uuid4()
    def __call__(self, updated_detection : dict, callback : typing.Callable = None):
        ret_events = []
        if 'is_moving' in updated_detection:
            if not self['is_moving'] and updated_detection['is_moving'] and updated_detection['estimated_movement_direction'] != 'Stopped':
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
        self._last_broadcast = time()

        self.plates : typing.Dict[str, dict] = {}
        
    async def start_task(self):
        await super().start_task()
        self.tm.app['rtc'].addInDataChannelHandler(**self.handlers)
        #self.handlers['log'].on('data', self._handleMessageLog)
        #self.handlers['log'].on('connect', self._handleLogSendAll)

    @staticmethod
    def check_detection_gate_intersection(det : Detection, det_gate : typing.Dict[str, np.ndarray]):
        box_data = {}
        pos = det['track_pos']
        if pos is None: pos = det['centroid_pos']
        x, y = pos

        for k, v in det_gate.items():
            box_data[k] = {'intersecting': False, 'state_change_ts': None}
            if v[0]<x<v[0]+v[2] and v[1]<y<v[1]+v[3]:
                box_data[k].update({'intersecting': True, 'state_change_ts': det['last_update_ts']})
        return box_data

    def _updateEntryExitState(self, det : Detection):
        gates = {
            'e2': np.array((1434, 113, 157, 959)),
            'e1': np.array((1600, 235, 199, 845))
        }
        if '_timings' not in det:
            det.update({'_timings': {x:{'_need_update': True} for x in gates.keys()}, "entry_exit_state": "-"})

        intersect_data = self.check_detection_gate_intersection(det, gates)

        e1_needs_update_check = det['_timings']['e1'].get('_need_update')
        e2_needs_update_check = det['_timings']['e2'].get('_need_update')

        if intersect_data['e1']['intersecting'] and e1_needs_update_check:
            det['_timings']['e1'].update(intersect_data['e1'])
            det['_timings']['e1'].update({'_need_update': False})
            print("Det", det['track_id'], "crossed gate e1")

        if intersect_data['e2']['intersecting'] and e2_needs_update_check:
            det['_timings']['e2'].update(intersect_data['e2'])
            det['_timings']['e2'].update({'_need_update': False})
            print("Det", det['track_id'], "crossed gate e2")

        if det['_timings']['e1'].get('intersecting') and det['_timings']['e2'].get('intersecting'):
            e1_timing = det['_timings']['e1'].get('state_change_ts')
            e2_timing = det['_timings']['e2'].get('state_change_ts')
            if e1_timing and e2_timing:
                delta : datetime.timedelta = (e2_timing - e1_timing)

                old_state = det.get('entry_exit_state')
                if delta > datetime.timedelta(0):
                    det.update({"entry_exit_state": "Entry"})
                else:
                    det.update({"entry_exit_state": "Exit"})

                new_state = det.get('entry_exit_state')
                if old_state != new_state:
                    ts = datetime.datetime.now()
                    if det['_ephemeral_plate'] in self.plates:
                        self.plates[det['_ephemeral_plate']].update({'last_state_ts': ts})
                    if new_state == 'Entry':
                        det.update({'entry_ts': ts})
                    elif new_state == 'Exit':
                        det.update({'exit_ts': ts})
                    print("Detection", det['track_id'], 'state changed to', new_state)

    async def __call__(self):
        while not self._stop.is_set():
            is_stop, next_items = await asyncio.gather(self._stop.wait_for(timeout=.1), self.save_queue.get_wait_for(timeout=.3))
            if is_stop: break
            if next_items is None: continue
            
            for batch in next_items:
                for det in batch:
                    if det['is_new_detection']:
                        new_det = Detection(det)
                        new_det.update({
                            "vehicle_type": TMP_CLASSID_TO_TYPE.get(new_det['detection_class'][0], "Vehicle")
                        })

                        new_ephimeral_plate_id = uuid.uuid4().hex
                        self.plates.update({
                            new_ephimeral_plate_id: {
                                'plate_number': '-',
                                'is_updated': True,
                                'is_active': True,
                                'last_state_ts': datetime.datetime.now(),
                                'associated_det': new_det
                            }
                        })
                        new_det.update({
                            "_ephemeral_plate": new_ephimeral_plate_id
                        })

                        self._submitPlateDecodeTask(new_det)

                        self._all_detections[det['track_id']] = new_det

                        msg = {
                            "title": "Vehicle detected",
                            "message": "New vehicle detected, recognising plate...",
                            "ts": new_det['first_detect_ts']
                        }
                        #self.handlers['log'].broadcast(msg)
                        #self._new_client_messages.append(msg)
                    else:
                        dtect : Detection = self._all_detections.get(det['track_id'])
                        if dtect is not None:
                            event = dtect(det) #Update with new detection frame
                            #if event:
                            #    self.handlers['log'].broadcast(event)
                            #    self._new_client_messages.extend(event)

                    dtect : Detection = self._all_detections.get(det['track_id'])
                    if dtect is not None:
                        self._updateEntryExitState(dtect)
            
            #Task progress check (periodically checked)
            for det in self._all_detections.values():
                self._checkPlateTask(det)
                
            self.save_queue.task_done()

            #Not efficient but for now its okay. Later needs to use delta and send on update
            await self._updateVehicleData()
            
    def _submitPlateDecodeTask(self, det : Detection):
        if self.plate_decoder is None: return
        self.plates[det['_ephemeral_plate']].update({
            'plate_number': "Detecting... [ID %d]" % det['track_id']
        })
        setattr(det, '_rec_task', asyncio.create_task(
            self.tm[self.plate_decoder].detectAndDecode(dict(det))
        ))

    def _checkPlateTask(self, det : Detection):
        rec_task : asyncio.Task = getattr(det, '_rec_task', None)
        if rec_task is not None:
            if rec_task.done():
                plate_detection = rec_task.result()
                print("Plate result:", plate_detection)
                #Bad detection, maybe try again
                if plate_detection is None:
                    #TODO: Bit delay in frames before retrying
                    #self._submitPlateDecodeTask(det)
                    delattr(det, '_rec_task')
                    return

                if plate_detection.get('code') < 0:
                    self.plates[det['_ephemeral_plate']].update({'plate_number': "Error %d" % plate_detection.get('code')})
                    #self.logger.debug("Plate number was not determined: %s", plate_detection.get('message'))
                    delattr(det, '_rec_task')
                    return
                
                event = det({"plate": plate_detection})

                #Check if this plate already exists
                plate_str = plate_detection['plate_str']
                if plate_str in self.plates:
                    #This plate was previously recognized and detected by a different detector.
                    #Make this detection also associated with it
                    del self.plates[det['_ephemeral_plate']]
                    self.plates[plate_str]['associated_det'] = det
                else:
                    #New plate detection (maybe)
                    self.plates[plate_str] = self.plates.pop(det['_ephemeral_plate'])
                    det['_ephemeral_plate'] = plate_str
                    self.plates[plate_str].update({
                        "plate_number": plate_str
                    })

                #if event:
                    #self.handlers['log'].broadcast(event)
                    #self._new_client_messages.extend(event)
                delattr(det, '_rec_task')

    #async def _handleMessageLog(self, msg, channel, peer):
    #    self.logger.info("Got message at channel %s: %s" % (channel.label, msg))

    #async def _handleLogSendAll(self, channel):
    #    channel.send(self._new_client_messages)

    async def _updateVehicleData(self):
        if time() - self._last_broadcast < 1.0:
            return
        vehicle_data = sorted([
            {
                "plate_number": y.get('plate_number') or 'Unknown',
                "type": y.get('associated_det', {}).get('vehicle_type', 'Unknown type'),
                "entry_exit_state": y.get('associated_det', {}).get('entry_exit_state', 'N/A'),
                "entry_state_ts": y.get('associated_det', {}).get("entry_ts", '-'),
                "exit_state_ts": y.get('associated_det', {}).get("exit_ts", '-'),
                "last_state_timestamp": y.get("last_state_ts"),
                "is_active": y.get("is_active", False)
            }
            for y in self.plates.values()
        ], key=lambda o: o['last_state_timestamp'], reverse=True)

        self.handlers['status'].broadcast(list(filter(lambda o: o['is_active'], vehicle_data)))
        self.handlers['log'].broadcast(vehicle_data)

        self._last_broadcast = time()

        #Mark as done (changes sent)
        for x in self.plates.values():
            x.update({"is_updated": False})

        '''self.handlers['status'].broadcast(sorted([
            {
                "track_id": det['track_id'],
                "plate_number": (det['plate']['plate_str'] or 'Unknown') if 'plate' in det else 'Detecting...',
                "detect_time": det.get('first_detect_ts')
            }
            for det in self._all_detections.values()
        ], key=lambda o: o['detect_time'], reverse=True))'''
