from enum import Enum

# the DRL will gather its rewards/punishments via checking these plane states
class PlaneState(Enum):
	AIR = 0
	LANDING = 1 # landing -> tells the DRL that the craft is landing IN THIS TICK
	WAITING_FOR_LANDING = 2 # waiting for landing -> once planes reach their final cruising altitude
	GROUND = 3 # grounded -> only used for planes in the queue
	TAKINGOFF = 4 # taking off -> tells the DRL that the craft is taking off IN THIS TICK
	WAITING_FOR_TAKEOFF = 5 # waiting for takeoff -> once planes are out on the runway
	CRASHED = 6