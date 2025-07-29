from hmac import new
from plane import Plane
from airport import *
from planestates import PlaneState

class PlaneManager:
    def __init__(self, max_slots=10):
        self.planes = []
        self.max_slots = max_slots
        self.id_to_callsign = [''] * max_slots  # id -> callsign
        self.callsign_to_id = {}               # callsign -> id

    def set_airport(self, airport: Airport):
        self.airport = airport

    def add_plane(self, init_state: dict):
        init_state['id'] = self.get_id(init_state['callsign'])

        new_plane = Plane(init_state)

        self.planes.append(new_plane)

        if init_state['state'] == PlaneState.QUEUED and self.airport:
            self.airport.queue.append((new_plane.id, self.airport.runways[27]))  # HACK: always use runway 27 for queued planes in this example
            #print(f"{new_plane.callsign} added to queue on runway 27 (ID: {new_plane.id})")
            #print(self.airport.queue)
        else:
            #print(f"{new_plane.callsign} added at ({new_plane.lat}, {new_plane.lon}) (ID: {new_plane.id})")
            pass

    # get a new id for a plane or return a plane's current id
    def get_id(self, callsign: str) -> int:
        if callsign in self.callsign_to_id:
            return self.callsign_to_id[callsign]

        for idx in range(self.max_slots):
            if self.id_to_callsign[idx] == '':
                self.id_to_callsign[idx] = callsign
                self.callsign_to_id[callsign] = idx
                return idx

        raise RuntimeError("No available plane slots.")

    # remove a plane from its id slot
    def delete_plane(self, id: int):
        if id < 0 or id >= self.max_slots:
            raise ValueError("Plane ID out of range.")
        
        if self.id_to_callsign[id] == '':
            raise ValueError(f"Plane with ID {id} does not exist.")
        self.callsign_to_id.pop(self.id_to_callsign[id])
        self.id_to_callsign[id] = ''

        idx_to_remove = next(idx for idx, plane in enumerate(self.planes) if plane.id == id)
        
        self.planes.pop(idx_to_remove)

    # remove a plane from its slot by callsign
    def delete_plane_by_callsign(self, callsign: str):
        if callsign not in self.callsign_to_id:
            raise ValueError(f"Flight {callsign} does not exist.")
        id = self.callsign_to_id.pop(callsign)
        self.id_to_callsign[id] = ''

        idx_to_remove = next(idx for idx, plane in enumerate(self.planes) if plane.id == id)

        self.planes.pop(idx_to_remove)

    # get the callsign associated with an id
    def get_callsign(self, slot_id: int) -> str:
        return self.id_to_callsign[slot_id]
    
    # show list of ids
    def show_ids(self):
        return self.id_to_callsign
    
    def print_planes(self, tick):
        for plane in self.planes:
            print(f"lat: {plane.lat}, lon: {plane.lon}, ID: {plane.id}, Callsign: {plane.callsign}, State: {plane.state}, command: {plane.command.command_type} tick: {tick}")

