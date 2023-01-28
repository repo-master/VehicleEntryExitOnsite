
from vehiclebot.task import AIOTask
from vehiclebot.management.rtc import DataChannelHandler

import typing

class VehicleData(AIOTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handlers : typing.Dict[str, DataChannelHandler] = dict(
            cmd = DataChannelHandler(),
            log = DataChannelHandler(),
            status = DataChannelHandler()
        )

        self._all_vehicles : set = set()
        self._entryexit_log = []

    async def start_task(self):
        self.tm.app['rtc'].addInDataChannelHandler(**self.handlers)

    async def __call__(self):
        self.on('state', self._updateVehicleData)
        self.on('broadcastChange', self.sendUpdateEvents)

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
            self._entryexit_log.append((vehicle, state, timestamp))
        elif state == "Exit":
            candidates = list(filter(self._filter_find_best_match_vehicle(vehicle), sorted(self._entryexit_log, key=lambda x: x[2], reverse=True)))
            if len(candidates) > 0:
                ee_log_entry_idx = self._entryexit_log.index(candidates[0])
                self.logger.info("Updating state of vehicle %s with state of %s as %s", self._entryexit_log[ee_log_entry_idx][0].id_short, vehicle.id_short, state)
                self._entryexit_log[ee_log_entry_idx] = (vehicle, state, timestamp)
            else:
                self._entryexit_log.append((vehicle, state, timestamp))
        await self.sendUpdateEvents()

    def generateDashboardData(self):
        vehicles_copy = self._all_vehicles.copy()
        log_copy = self._entryexit_log.copy()

        vehicle_data = sorted([
            {
                "id": y.id_short,
                "plate_number": y.license_plate['plate_str'] or '---',
                "type": '',
                "entry_exit_state": y.state,
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
                    "type": '',
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

        return vehicle_data, log_data

    async def sendUpdateEvents(self, vehicles : dict = None):
        if vehicles is not None:
            self._all_vehicles = vehicles

        vehicle_data, log_data = self.generateDashboardData()

        self.handlers['status'].broadcast(vehicle_data)
        self.handlers['log'].broadcast(log_data)
