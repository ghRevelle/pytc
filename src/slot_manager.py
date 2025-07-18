class FixedSlotPlaneManager:
	# assigning planes and their callsigns to a unique integer ID with a limited number of ID slots
    def __init__(self, max_slots=10):
        self.max_slots = max_slots
        self.slot_to_callsign = [None] * max_slots  # index: ID, value: callsign or None
        self.callsign_to_slot = {}                  # callsign -> slot index

    def get_id(self, callsign: str) -> int:
        if callsign in self.callsign_to_slot:
            return self.callsign_to_slot[callsign]

        for idx in range(self.max_slots):
            if self.slot_to_callsign[idx] is None:
                self.slot_to_callsign[idx] = callsign
                self.callsign_to_slot[callsign] = idx
                return idx

        raise RuntimeError("No available plane slots.")

    def release_slot(self, callsign: str):
        if callsign not in self.callsign_to_slot:
            raise ValueError(f"{callsign} not found in active slots.")
        idx = self.callsign_to_slot.pop(callsign)
        self.slot_to_callsign[idx] = None

    def get_slot(self, callsign: str) -> int:
        return self.callsign_to_slot.get(callsign, -1)

    def get_callsign(self, slot_id: int) -> str:
        return self.slot_to_callsign[slot_id]
