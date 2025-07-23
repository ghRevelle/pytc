from enum import Enum

# the DRL will gather its rewards/punishments via checking these plane states
class PlaneState(Enum):
	AIR = 0
	LANDING = 1 # landing -> tells the DRL that the craft is landing IN THIS TICK
	TAKINGOFF = 2 # taking off -> tells the DRL that the craft is taking off IN THIS TICK
	GROUND = 3 # grounded -> only used for planes in the queue
	WAITING_FOR_TAKEOFF = 4 # waiting for takeoff -> used to make sure a plane is ready to be cleared for takeoff
	CRASHED = 5