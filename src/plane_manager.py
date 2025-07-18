from plane import Plane

class PlaneManager:
    def __init__(self, plane_infos=[], max_slots=10):
        self.planes = []
        self.max_slots = max_slots
        self.id_to_callsign = [''] * max_slots  # id -> callsign
        self.callsign_to_id = {}               # callsign -> id
        for plane_info in plane_infos:
            if not isinstance(plane_info, dict):
                raise TypeError("Plane info must be a dictionary.")
            self.add_plane(plane_info)

    def add_plane(self, init_state: dict):
        init_state['id'] = self.get_id(init_state['callsign'])
        self.planes.append(Plane(init_state))

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

        for idx in range(len(self.planes)):
            if self.planes[idx].id == id:
                idx_to_remove = idx
        self.planes.pop(idx_to_remove)

    # remove a plane from its slot by callsign
    def delete_plane_by_callsign(self, callsign: str):
        if callsign not in self.callsign_to_id:
            raise ValueError(f"Flight {callsign} does not exist.")
        idx = self.callsign_to_id.pop(callsign)
        self.id_to_callsign[idx] = ''

        for idx in range(len(self.planes)):
            if self.planes[idx].id == id:
                idx_to_remove = idx
        self.planes.pop(idx_to_remove)

    # get the callsign associated with an id
    def get_callsign(self, slot_id: int) -> str:
        return self.id_to_callsign[slot_id]
    
    # show list of ids
    def show_ids(self):
        return self.id_to_callsign
