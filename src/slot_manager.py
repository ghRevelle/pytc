class FixedSlotPlaneManager:
	# assigning planes and their callsigns to a unique integer ID with a limited number of ID slots
    def __init__(self, max_slots=10):
        self.max_slots = max_slots
        self.id_to_callsign = [''] * max_slots  # id -> callsign
        self.callsign_to_id = {}                  # callsign -> id

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
    def release_id(self, slot_id: int):
        if slot_id < 0 or slot_id >= self.max_slots:
            raise ValueError("ID out of range.")
        
        if self.id_to_callsign[slot_id] == '':
            raise ValueError(f"Slot {slot_id} is empty.")
        self.callsign_to_id.pop(self.id_to_callsign[slot_id])
        self.id_to_callsign[slot_id] = ''

    # remove a plane from its slot by callsign
    def release_id_by_callsign(self, callsign: str):
        if callsign not in self.callsign_to_id:
            raise ValueError(f"{callsign} not found in active slots.")
        idx = self.callsign_to_id.pop(callsign)
        self.id_to_callsign[idx] = ''

    # get the callsign associated with an id
    def get_callsign(self, slot_id: int) -> str:
        return self.id_to_callsign[slot_id]
    
    # show list of ids
    def show_ids(self):
        return self.id_to_callsign
