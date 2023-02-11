
from vehiclebot.task import AIOTask
from vehiclebot.management.rtc import DataChannelHandler
from aiortc import RTCDataChannel

import datetime
import typing

class VehicleData(AIOTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handlers : typing.Dict[str, DataChannelHandler] = dict(
            cmd = DataChannelHandler(),
            log = DataChannelHandler(),
            status = DataChannelHandler()
        )

        #
        self._all_vehicles : set = set()
        self._entryexit_log = []

    async def start_task(self):
        self.tm.app['rtc'].addInDataChannelHandler(**self.handlers)

    async def __call__(self):
        self.on('state', self._updateVehicleData)
        self.on('broadcastChange', self.sendUpdateEvents)
        self.handlers['log'].on('connect', self._sendNewConnectionLogData)
        self.handlers['status'].on('connect', self._sendNewConnectionStatusData)

    def calculate_duration(self, v):
        entry_ts = v.timings['last_entry']
        exit_ts = v.timings['last_exit']
        if entry_ts is not None and exit_ts is not None and exit_ts > entry_ts:
            return exit_ts - entry_ts

    def _filter_find_best_match_vehicle(self, vehicle):
        def _wrap(v_entry_exit_log_item):
            if v_entry_exit_log_item[0].license_plate.plate_known and vehicle.license_plate.plate_known:
                return (v_entry_exit_log_item[0].license_plate.matchesWith(vehicle.license_plate))
            return v_entry_exit_log_item[0].id == vehicle.id
        return _wrap

    async def _updateVehicleData(self, vehicle, state, timestamp):
        if state == "Entry":
            #New entry vehicle, so create new log row
            self._entryexit_log.append((vehicle, state, timestamp))
        elif state == "Exit":
            #Exit can belong to an already entered vehicle, so try to update the same vehicle that had entered
            #otherwise create a new log row
            candidates = list(filter(self._filter_find_best_match_vehicle(vehicle), sorted(self._entryexit_log, key=lambda x: x[2], reverse=True)))
            if len(candidates) > 0:
                ee_log_entry_idx = self._entryexit_log.index(candidates[0])
                self.logger.info("Updating state of vehicle %s with state of %s as %s", self._entryexit_log[ee_log_entry_idx][0].id_short, vehicle.id_short, state)
                self._entryexit_log[ee_log_entry_idx] = (vehicle, state, timestamp)
            else:
                self._entryexit_log.append((vehicle, state, timestamp))
        #Update on client
        await self.sendUpdateEvents()

    def generateDashboardData(self):
        '''Create data lists that correspond to the active vehicles and log entries'''
        vehicles_copy = self._all_vehicles.copy()
        log_copy = self._entryexit_log.copy()

        vehicle_data = sorted([
            {
                "id": y.id_short,
                "plate_number": y.license_plate['plate_str'] or '---',
                "type": 'Car',
                "entry_exit_state": y.state or '---',
                "last_state_timestamp": y.state_ts or y.first_detected_ts
            }
            for y in vehicles_copy if y.is_active
        ], key=lambda o: (o['last_state_timestamp'] is None, o['last_state_timestamp']), reverse=True)

        log_data = sorted(
            map(
                lambda e: {
                    "id": e[0].id_short,
                    "event_ts": e[2],
                    "plate_number": e[0].license_plate['plate_str'] or '---',
                    "type": 'Car',
                    "entry_exit_state": e[1],
                    "entry_state_ts": e[0].timings['last_entry'],
                    "exit_state_ts": e[0].timings['last_exit'],
                    "last_state_timestamp": e[0].state_ts,
                    "presence_duration": self.calculate_duration(e[0]),
                },
                log_copy
            ),
            key=lambda o: (o['last_state_timestamp'] is None, o['last_state_timestamp'], o['plate_number']),
            reverse=True
        )

        log_data.extend([
            {
                "id": 0,
                "event_ts": datetime.datetime(2023,2,6,12,44,10),
                "plate_number": 'GA 01 K 1021',
                "type": 'Car',
                "entry_exit_state": 'Exit',
                "entry_state_ts": datetime.datetime(2023,2,6,9,27,40),
                "exit_state_ts": datetime.datetime(2023,2,6,12,44,10),
                "last_state_timestamp": datetime.datetime(2023,2,6,12,44,10),
                "presence_duration": datetime.datetime(2023,2,6,12,44,10) - datetime.datetime(2023,2,6,9,27,40),
            },
            {
                "id": 1,
                "event_ts": datetime.datetime(2023,2,6,16,30,45),
                "plate_number": 'MH 12 AH 7473',
                "type": 'Car',
                "entry_exit_state": 'Exit',
                "entry_state_ts": datetime.datetime(2023,2,6,9,32,10),
                "exit_state_ts": datetime.datetime(2023,2,6,16,30,45),
                "last_state_timestamp": datetime.datetime(2023,2,6,16,30,45),
                "presence_duration": datetime.datetime(2023,2,6,16,30,45) - datetime.datetime(2023,2,6,9,32,10),
            },
            {
                "id": 2,
                "event_ts": datetime.datetime(2023,2,6,12,10,33),
                "plate_number": 'GA 01 AE 3304',
                "type": 'Car',
                "entry_exit_state": 'Entry',
                "entry_state_ts": datetime.datetime(2023,2,6,12,10,33),
                "exit_state_ts": None,
                "last_state_timestamp": datetime.datetime(2023,2,6,12,10,33),
                "presence_duration": None,
            }
        ])

        return vehicle_data, log_data

    async def sendUpdateEvents(self, vehicles : dict = None):
        '''Send status and log entries to all connected clients'''
        if vehicles is not None:
            self._all_vehicles = vehicles

        vehicle_data, log_data = self.generateDashboardData()

        self.handlers['status'].broadcast(vehicle_data)
        self.handlers['log'].broadcast(log_data)

    #Routines to send data to newly connected clients

    async def _sendNewConnectionLogData(self, log_channel : RTCDataChannel):
        _, log_data = self.generateDashboardData()
        log_channel.send(log_data)

    async def _sendNewConnectionStatusData(self, status_channel : RTCDataChannel):
        vehicle_data, _ = self.generateDashboardData()
        status_channel.send(vehicle_data)
